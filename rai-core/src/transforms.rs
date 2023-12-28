use std::{
    collections::{BTreeSet, HashMap},
    mem::MaybeUninit,
};

use tracing::Level;

use crate::{dispatch::eval_rule, utils::TensorIter, Tensor};
pub trait Function<const IN: usize, const OUT: usize> {
    fn call(&self, primals: &[Tensor; IN]) -> [Tensor; OUT];
}

impl<F> Function<1, 1> for F
where
    F: Fn(&Tensor) -> Tensor,
{
    fn call(&self, primals: &[Tensor; 1]) -> [Tensor; 1] {
        [self(&primals[0])]
    }
}

impl<F> Function<2, 1> for F
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    fn call(&self, primals: &[Tensor; 2]) -> [Tensor; 1] {
        [self(&primals[0], &primals[1])]
    }
}

impl<F> Function<3, 1> for F
where
    F: Fn(&Tensor, &Tensor, &Tensor) -> Tensor,
{
    fn call(&self, primals: &[Tensor; 3]) -> [Tensor; 1] {
        [self(&primals[0], &primals[1], &primals[2])]
    }
}

pub fn jvp<const IN: usize, const OUT: usize, F>(
    func: F,
    primals: &[Tensor; IN],
    tangents: &[Tensor; IN],
) -> ([Tensor; OUT], [Tensor; OUT])
where
    F: Function<IN, OUT>,
{
    let mut tangent_map = HashMap::new();

    for (p, t) in primals.iter().zip(tangents) {
        tangent_map.insert(p.id(), t.clone());
    }
    let outputs = func.call(primals);

    let mut jvps: [MaybeUninit<Tensor>; OUT] = unsafe { MaybeUninit::uninit().assume_init() };
    for (i, t) in outputs.iter().enumerate() {
        jvps[i] = MaybeUninit::new(t.jvp(&mut tangent_map));
    }
    let jvps = unsafe { MaybeUninit::array_assume_init(jvps) };

    (outputs, jvps)
}

type VjpFn<const IN: usize, const OUT: usize> = Box<dyn Fn([Tensor; OUT]) -> [Tensor; IN]>;

#[tracing::instrument(skip(func), name = "transform::vjp" level = Level::TRACE)]
pub fn vjp<const IN: usize, const OUT: usize, F>(
    func: F,
    primals: &[Tensor; IN],
) -> ([Tensor; OUT], VjpFn<IN, OUT>)
where
    F: Function<IN, OUT>,
{
    let outputs = func.call(primals);

    let outs = outputs.clone();
    let primal_ids: Vec<usize> = primals.iter().map(|v| v.id()).collect();
    let vjps_fn = move |cotangents: [Tensor; OUT]| {
        let mut cotangent_map = HashMap::new();

        for (p, c) in outs.iter().zip(cotangents) {
            cotangent_map.insert(p.id(), c);
        }

        for t in outs.iter() {
            t.vjp(&mut cotangent_map);
        }

        let mut vjps: [MaybeUninit<Tensor>; IN] = unsafe { MaybeUninit::uninit().assume_init() };

        for (i, id) in primal_ids.iter().enumerate() {
            vjps[i] = MaybeUninit::new(cotangent_map.get(id).unwrap().clone());
        }

        unsafe { MaybeUninit::array_assume_init(vjps) }
    };

    (outputs, Box::new(vjps_fn))
}

#[derive(Clone)]
pub struct GradFunc<const IN: usize, const OUT: usize, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    func: F,
}
impl<const IN: usize, const OUT: usize, F> GradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }

    #[tracing::instrument(skip(self), name = "GradFunc::call" level = Level::TRACE)]
    fn call(&self, primals: &[Tensor; IN]) -> [Tensor; IN] {
        let (outputs, f_vjp) = vjp(self.func.clone(), primals);

        let mut cotangents: [MaybeUninit<Tensor>; OUT] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for (i, t) in outputs.iter().enumerate() {
            cotangents[i] = MaybeUninit::new(t.ones_like());
        }

        let cotangents = unsafe { MaybeUninit::array_assume_init(cotangents) };

        f_vjp(cotangents)
    }
}

impl<const IN: usize, const OUT: usize, F> FnOnce<(&[Tensor; IN],)> for GradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    type Output = [Tensor; IN];

    extern "rust-call" fn call_once(self, args: (&[Tensor; IN],)) -> Self::Output {
        let inputs = args.0;
        self.call(inputs)
    }
}

impl<const IN: usize, const OUT: usize, F> FnMut<(&[Tensor; IN],)> for GradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    extern "rust-call" fn call_mut(&mut self, args: (&[Tensor; IN],)) -> Self::Output {
        let inputs = args.0;
        self.call(inputs)
    }
}

impl<const IN: usize, const OUT: usize, F> Fn<(&[Tensor; IN],)> for GradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    extern "rust-call" fn call(&self, args: (&[Tensor; IN],)) -> Self::Output {
        let inputs = args.0;
        self.call(inputs)
    }
}

impl<const IN: usize, const OUT: usize, F> Function<IN, IN> for GradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    fn call(&self, primals: &[Tensor; IN]) -> [Tensor; IN] {
        self(primals)
    }
}

pub struct ValueAndGradFunc<const IN: usize, const OUT: usize, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    func: F,
}

impl<const IN: usize, const OUT: usize, F> ValueAndGradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }

    fn call(&self, primals: &[Tensor; IN]) -> ([Tensor; OUT], [Tensor; IN]) {
        let (outputs, f_vjp) = vjp(self.func.clone(), primals);

        let mut cotangents: [MaybeUninit<Tensor>; OUT] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for (i, t) in outputs.iter().enumerate() {
            cotangents[i] = MaybeUninit::new(t.ones_like());
        }

        let cotangents = unsafe { MaybeUninit::array_assume_init(cotangents) };

        let grads = f_vjp(cotangents);
        (outputs, grads)
    }
}

impl<const IN: usize, const OUT: usize, F> FnOnce<(&[Tensor; IN],)> for ValueAndGradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    type Output = ([Tensor; OUT], [Tensor; IN]);

    extern "rust-call" fn call_once(self, args: (&[Tensor; IN],)) -> Self::Output {
        let inputs = args.0;
        self.call(inputs)
    }
}

impl<const IN: usize, const OUT: usize, F> FnMut<(&[Tensor; IN],)> for ValueAndGradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    extern "rust-call" fn call_mut(&mut self, args: (&[Tensor; IN],)) -> Self::Output {
        let inputs = args.0;
        self.call(inputs)
    }
}

impl<const IN: usize, const OUT: usize, F> Fn<(&[Tensor; IN],)> for ValueAndGradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    extern "rust-call" fn call(&self, args: (&[Tensor; IN],)) -> Self::Output {
        let inputs = args.0;
        self.call(inputs)
    }
}

#[tracing::instrument(skip(func), level = Level::TRACE)]
pub fn grad<const IN: usize, const OUT: usize, F>(func: F) -> GradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    GradFunc::new(func)
}

pub fn value_and_grad<const IN: usize, const OUT: usize, F>(func: F) -> ValueAndGradFunc<IN, OUT, F>
where
    F: Function<IN, OUT> + Clone + 'static,
{
    ValueAndGradFunc::new(func)
}

// TODO: args with retain_graph?
pub fn eval<T: TensorIter>(args: T) {
    let mut tape = BTreeSet::new();
    for output in args.tensor_iter() {
        topological_sort(&mut tape, output);
    }
    for t in tape.into_iter() {
        {
            let backend = t.backend();
            let primitive = t.primitive();
            let inputs = &*t.inputs();
            let rule = eval_rule(backend, primitive).unwrap_or_else(|| {
                panic!(
                    "no eval rule for backend: {:?}, primitive: {:?}",
                    backend, primitive
                )
            });
            rule.eval(backend, primitive, inputs, &t);
        }
        t.detach();
    }
}

fn topological_sort(tape: &mut BTreeSet<Tensor>, t: &Tensor) {
    for input in t.inputs().iter() {
        if !t.is_evalualted() {
            topological_sort(tape, input);
        }
    }
    if t.is_evalualted() || tape.contains(t) {
        return;
    }
    tape.insert(t.clone());
}
