use crate::dispatch::eval_rule;
use crate::utils::TensorIter;
use crate::{Module, Shape, Tensor};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;

pub trait Func<IN, OUT>
where
    IN: WithTensors,
    OUT: WithTensors,
{
    type Tangent: ToTensorGrads + FromTensorGrads;
    type Cotangent: ToTensorGrads + FromTensorGrads;
    fn call(&self, input: IN) -> OUT;
    fn self_captured_tensors(&self, _tensors: &mut Vec<Tensor>) {}
    fn extract_input_tensors(&self, input: &IN, tensors: &mut Vec<Tensor>);
}

pub trait WithTensors {
    fn tensors(&self) -> Vec<Tensor>;
}

impl<const N: usize> WithTensors for [Tensor; N] {
    fn tensors(&self) -> Vec<Tensor> {
        self.to_vec()
    }
}

impl<const N: usize> WithTensors for &[Tensor; N] {
    fn tensors(&self) -> Vec<Tensor> {
        self.to_vec()
    }
}

impl WithTensors for Vec<Tensor> {
    fn tensors(&self) -> Vec<Tensor> {
        self.clone()
    }
}

impl WithTensors for Tensor {
    fn tensors(&self) -> Vec<Tensor> {
        vec![self.clone()]
    }
}

impl WithTensors for &Tensor {
    fn tensors(&self) -> Vec<Tensor> {
        vec![(*self).clone()]
    }
}

impl WithTensors for (Tensor, Tensor) {
    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone(), self.1.clone()]
    }
}

impl WithTensors for HashMap<usize, Tensor> {
    fn tensors(&self) -> Vec<Tensor> {
        self.values().cloned().collect()
    }
}

impl<M> WithTensors for (&M, Tensor)
where
    M: Module,
{
    fn tensors(&self) -> Vec<Tensor> {
        self.0.parameters()
    }
}

impl<M> WithTensors for (&M, &Tensor)
where
    M: Module,
{
    fn tensors(&self) -> Vec<Tensor> {
        self.0.parameters()
    }
}

impl<M> WithTensors for (&M, Tensor, Tensor)
where
    M: Module,
{
    fn tensors(&self) -> Vec<Tensor> {
        self.0.parameters()
    }
}

impl<M> WithTensors for (&M, &Tensor, &Tensor)
where
    M: Module,
{
    fn tensors(&self) -> Vec<Tensor> {
        self.0.parameters()
    }
}

pub trait FromTensorGrads {
    fn from_tensor_grads(tensors: &[Tensor], grads: HashMap<usize, Tensor>) -> Self;
}

impl<const N: usize> FromTensorGrads for [Tensor; N] {
    fn from_tensor_grads(tensors: &[Tensor], mut grads: HashMap<usize, Tensor>) -> Self {
        tensors
            .iter()
            .map(|t| grads.remove(&t.id()).unwrap())
            .collect::<Vec<Tensor>>()
            .try_into()
            .unwrap_or_else(|v: Vec<Tensor>| {
                panic!("Expected a Vec of length {} but it was {}", N, v.len())
            })
    }
}

impl FromTensorGrads for HashMap<usize, Tensor> {
    fn from_tensor_grads(tensors: &[Tensor], grads: HashMap<usize, Tensor>) -> Self {
        debug_assert!(tensors.len() == grads.len());
        // TODO: check id list match?
        grads
    }
}

pub trait ToTensorGrads {
    fn to_tensor_grads(self, tensors: &[Tensor]) -> HashMap<usize, Tensor>;
}

impl ToTensorGrads for HashMap<usize, Tensor> {
    fn to_tensor_grads(self, tensors: &[Tensor]) -> HashMap<usize, Tensor> {
        debug_assert!(self.len() == tensors.len());
        self
    }
}

impl<const N: usize> ToTensorGrads for [Tensor; N] {
    fn to_tensor_grads(self, tensors: &[Tensor]) -> HashMap<usize, Tensor> {
        self.into_iter()
            .enumerate()
            .map(|(i, t)| (tensors[i].id(), t))
            .collect()
    }
}

pub fn jvp<IN, OUT, F>(func: F, input: IN, tangents: F::Tangent) -> (OUT, F::Cotangent)
where
    F: Func<IN, OUT>,
    IN: WithTensors,
    OUT: WithTensors,
{
    let mut input_tensors = Vec::new();
    func.self_captured_tensors(&mut input_tensors);
    func.extract_input_tensors(&input, &mut input_tensors);
    let mut tangent_map = tangents.to_tensor_grads(&input_tensors);
    let output = func.call(input);
    let out_tensors = output.tensors();
    let mut jvps = HashMap::new();
    for t in &out_tensors {
        jvps.insert(t.id(), t.jvp(&mut tangent_map));
    }
    let jvps = F::Cotangent::from_tensor_grads(&out_tensors, jvps);
    (output, jvps)
}

type VjpFn<IN, OUT> = Box<dyn Fn(IN) -> OUT>;

pub fn vjp<IN, OUT, F>(func: F, input: IN) -> (OUT, VjpFn<F::Cotangent, F::Tangent>)
where
    F: Func<IN, OUT>,
    IN: WithTensors,
    OUT: WithTensors,
{
    let mut input_tensors = Vec::new();
    func.self_captured_tensors(&mut input_tensors);
    func.extract_input_tensors(&input, &mut input_tensors);
    let output = func.call(input);
    let out_tensors = output.tensors();
    let vjps_fn = move |cotangents: F::Cotangent| {
        let mut cotangent_map = cotangents.to_tensor_grads(&out_tensors);
        for t in out_tensors.iter() {
            t.vjp(&mut cotangent_map);
        }
        let mut vjps = HashMap::new();
        for t in input_tensors.iter() {
            let id = t.id();
            let c = cotangent_map.get(&id).unwrap().clone();
            if c.shape_eq(t) {
                vjps.insert(id, c);
            } else {
                // reduce sum cotangent to its input shape
                // TODO: will cotangent ndim always > input ndim?
                vjps.insert(id, c.sum(..t.ndim()));
            }
        }
        F::Tangent::from_tensor_grads(&input_tensors, vjps)
    };
    (output, Box::new(vjps_fn))
}

#[derive(Clone)]
pub struct GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    func: F,
    phantom: std::marker::PhantomData<(IN, OUT)>,
}

impl<IN, OUT, F> GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }

    fn apply(&self, input: IN) -> F::Tangent {
        let (output, vjp_fn) = vjp(self.func.clone(), input);
        let mut cotagents = HashMap::new();
        let out_tensors = output.tensors();
        for t in &out_tensors {
            cotagents.insert(t.id(), t.ones_like());
        }
        vjp_fn(F::Cotangent::from_tensor_grads(&out_tensors, cotagents))
    }
}

impl<IN, OUT, F> Func<IN, OUT> for GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    type Tangent = F::Tangent;
    type Cotangent = F::Cotangent;
    fn call(&self, input: IN) -> OUT {
        self.func.call(input)
    }

    fn self_captured_tensors(&self, tensors: &mut Vec<Tensor>) {
        self.func.self_captured_tensors(tensors)
    }

    fn extract_input_tensors(&self, input: &IN, tensors: &mut Vec<Tensor>) {
        self.func.extract_input_tensors(input, tensors)
    }
}

impl<IN, OUT, F> FnOnce<(IN,)> for GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    type Output = F::Tangent;
    extern "rust-call" fn call_once(self, args: (IN,)) -> Self::Output {
        self.apply(args.0)
    }
}

impl<IN, OUT, F> FnMut<(IN,)> for GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    extern "rust-call" fn call_mut(&mut self, args: (IN,)) -> Self::Output {
        self.apply(args.0)
    }
}

impl<IN, OUT, F> Fn<(IN,)> for GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    extern "rust-call" fn call(&self, args: (IN,)) -> Self::Output {
        self.apply(args.0)
    }
}

pub fn grad<IN, OUT, F>(func: F) -> GradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    GradFunc::new(func)
}

#[derive(Clone)]
pub struct ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    func: F,
    phantom: std::marker::PhantomData<(IN, OUT)>,
}

impl<IN, OUT, F> ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }

    fn apply(&self, input: IN) -> (OUT, F::Tangent) {
        let (output, vjp_fn) = vjp(self.func.clone(), input);
        let mut cotagents = HashMap::new();
        let out_tensors = output.tensors();
        for t in &out_tensors {
            cotagents.insert(t.id(), t.ones_like());
        }
        let tangents = vjp_fn(F::Cotangent::from_tensor_grads(&out_tensors, cotagents));
        (output, tangents)
    }
}

impl<IN, OUT, F> Func<IN, OUT> for ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    type Tangent = F::Tangent;
    type Cotangent = F::Cotangent;
    fn call(&self, input: IN) -> OUT {
        self.func.call(input)
    }

    fn self_captured_tensors(&self, tensors: &mut Vec<Tensor>) {
        self.func.self_captured_tensors(tensors)
    }

    fn extract_input_tensors(&self, input: &IN, tensors: &mut Vec<Tensor>) {
        self.func.extract_input_tensors(input, tensors)
    }
}

impl<IN, OUT, F> FnOnce<(IN,)> for ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    type Output = (OUT, F::Tangent);
    extern "rust-call" fn call_once(self, args: (IN,)) -> Self::Output {
        self.apply(args.0)
    }
}

impl<IN, OUT, F> FnMut<(IN,)> for ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    extern "rust-call" fn call_mut(&mut self, args: (IN,)) -> Self::Output {
        self.apply(args.0)
    }
}

impl<IN, OUT, F> Fn<(IN,)> for ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    extern "rust-call" fn call(&self, args: (IN,)) -> Self::Output {
        self.apply(args.0)
    }
}

pub fn value_and_grad<IN, OUT, F>(func: F) -> ValueAndGradFunc<IN, OUT, F>
where
    F: Func<IN, OUT> + Clone + 'static,
    IN: WithTensors,
    OUT: WithTensors,
{
    ValueAndGradFunc::new(func)
}

// impl for Fn
impl<F> Func<[Tensor; 1], [Tensor; 1]> for F
where
    F: Fn(&Tensor) -> Tensor,
{
    type Tangent = [Tensor; 1];
    type Cotangent = [Tensor; 1];
    fn call(&self, input: [Tensor; 1]) -> [Tensor; 1] {
        [self(&input[0])]
    }

    fn extract_input_tensors(&self, input: &[Tensor; 1], inputs: &mut Vec<Tensor>) {
        inputs.extend(input.iter().cloned());
    }
}

impl<F> Func<[Tensor; 2], [Tensor; 1]> for F
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    type Tangent = [Tensor; 2];
    type Cotangent = [Tensor; 1];
    fn call(&self, input: [Tensor; 2]) -> [Tensor; 1] {
        [self(&input[0], &input[1])]
    }

    fn extract_input_tensors(&self, input: &[Tensor; 2], inputs: &mut Vec<Tensor>) {
        inputs.extend(input.iter().cloned());
    }
}

impl<F> Func<[Tensor; 3], [Tensor; 1]> for F
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor,
{
    type Tangent = [Tensor; 3];
    type Cotangent = [Tensor; 1];
    fn call(&self, input: [Tensor; 3]) -> [Tensor; 1] {
        [self(&input[0], &input[1], &input[2])]
    }

    fn extract_input_tensors(&self, input: &[Tensor; 3], inputs: &mut Vec<Tensor>) {
        inputs.extend(input.iter().cloned());
    }
}

pub trait EvalArgs: Debug {
    fn outputs(&self) -> impl Iterator<Item = &Tensor>;
    fn retain_graph(&self) -> bool {
        false
    }
}
impl<T> EvalArgs for T
where
    T: TensorIter,
{
    fn outputs(&self) -> impl Iterator<Item = &Tensor> {
        self.tensor_iter()
    }
}

impl<T> EvalArgs for (T, bool)
where
    T: TensorIter,
{
    fn outputs(&self) -> impl Iterator<Item = &Tensor> {
        self.0.tensor_iter()
    }

    fn retain_graph(&self) -> bool {
        self.1
    }
}

pub fn eval<T: EvalArgs>(args: T) {
    let mut tape = BTreeSet::new();
    for output in args.outputs() {
        topological_sort(&mut tape, output);
    }
    for t in tape.into_iter() {
        {
            let backend = t.backend().clone_boxed();
            let backend = backend.as_ref();
            let primitive = t.primitive().clone_boxed();
            let primitive = primitive.as_ref();
            let inputs = &*t.inputs();
            let rule = eval_rule(backend, primitive).unwrap_or_else(|| {
                panic!(
                    "no eval rule for backend: {:?}, primitive: {:?}",
                    backend, primitive
                )
            });
            rule.eval(backend, primitive, inputs, &t);
        }
        if !args.retain_graph() {
            t.detach();
        }
    }
}

fn topological_sort(tape: &mut BTreeSet<Tensor>, t: &Tensor) {
    for input in t.inputs().iter() {
        if !t.is_evaluated() {
            topological_sort(tape, input);
        }
    }
    if t.is_evaluated() || tape.contains(t) {
        return;
    }
    tape.insert(t.clone());
}
