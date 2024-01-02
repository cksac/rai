use std::{any::TypeId, collections::HashMap, sync::Mutex};

use dyn_clone::DynClone;
use once_cell::sync::Lazy;

use crate::{backend::Cpu, primitives, Backend, Primitive, Tensor};

pub trait Eval<B, P>: DynClone + Sync + Send + 'static
where
    B: ?Sized,
    P: ?Sized,
{
    fn eval(&self, backend: &B, primitive: &P, inputs: &[Tensor], output: &Tensor);
}
dyn_clone::clone_trait_object!(<B, P> Eval<B, P> where B: ?Sized, P: ?Sized);

#[derive(Debug, Clone)]
pub struct Dispatch<B, P> {
    phantom: std::marker::PhantomData<(B, P)>,
}

impl<B, P> Eval<dyn Backend, dyn Primitive> for Dispatch<B, P>
where
    B: Backend + 'static + Sync + Send,
    P: Primitive + 'static + Sync + Send,
    Self: Eval<B, P> + 'static,
{
    #[inline]
    fn eval(
        &self,
        backend: &dyn Backend,
        primitive: &dyn Primitive,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let backend = backend.as_any().downcast_ref::<B>().unwrap();
        let primitive = primitive.as_any().downcast_ref::<P>().unwrap();
        self.eval(backend, primitive, inputs, output);
    }
}

impl<B, P> Eval<Box<dyn Backend>, Box<dyn Primitive>> for Dispatch<B, P>
where
    B: Backend + 'static + Sync + Send,
    P: Primitive + 'static + Sync + Send,
    Self: Eval<B, P> + 'static,
{
    #[inline]
    fn eval(
        &self,
        backend: &Box<dyn Backend>,
        primitive: &Box<dyn Primitive>,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let backend = backend.as_any().downcast_ref::<B>().unwrap();
        let primitive = primitive.as_any().downcast_ref::<P>().unwrap();
        self.eval(backend, primitive, inputs, output);
    }
}

type ErasedEval = Box<dyn Eval<dyn Backend, dyn Primitive>>;

static EVAL_DISPATCHER: Lazy<Mutex<HashMap<(TypeId, TypeId), ErasedEval>>> = Lazy::new(|| {
    let mut rules: HashMap<(TypeId, TypeId), ErasedEval> = HashMap::new();

    // creation
    _register::<Cpu, primitives::Full>(&mut rules);
    _register::<Cpu, primitives::Normal>(&mut rules);
    _register::<Cpu, primitives::Arange>(&mut rules);
    // binary
    _register::<Cpu, primitives::Add>(&mut rules);
    _register::<Cpu, primitives::Sub>(&mut rules);
    _register::<Cpu, primitives::Mul>(&mut rules);
    _register::<Cpu, primitives::Div>(&mut rules);
    _register::<Cpu, primitives::MatMul>(&mut rules);
    // unary
    _register::<Cpu, primitives::Sin>(&mut rules);
    _register::<Cpu, primitives::Cos>(&mut rules);
    _register::<Cpu, primitives::Negative>(&mut rules);
    _register::<Cpu, primitives::Square>(&mut rules);
    _register::<Cpu, primitives::Sqrt>(&mut rules);
    _register::<Cpu, primitives::Rsqrt>(&mut rules);
    _register::<Cpu, primitives::Sign>(&mut rules);
    _register::<Cpu, primitives::Abs>(&mut rules);
    _register::<Cpu, primitives::Exp>(&mut rules);
    // transform
    _register::<Cpu, primitives::Transpose>(&mut rules);
    _register::<Cpu, primitives::Reshape>(&mut rules);
    _register::<Cpu, primitives::Broadcast>(&mut rules);
    // reduce
    _register::<Cpu, primitives::ReduceSum>(&mut rules);

    Mutex::new(rules)
});

pub fn register<B, P>()
where
    B: Backend + 'static + Sync + Send,
    P: Primitive + 'static + Sync + Send,
    Dispatch<B, P>: Eval<B, P> + 'static,
{
    let mut rules = EVAL_DISPATCHER.lock().unwrap();
    _register::<B, P>(&mut rules)
}

pub fn register_custom<B, P, R>(rule: R)
where
    B: Backend + 'static + Sync + Send,
    P: Primitive + 'static + Sync + Send,
    R: Eval<B, P> + 'static + Eval<dyn Backend, dyn Primitive>,
{
    let mut rules = EVAL_DISPATCHER.lock().unwrap();
    rules.insert((TypeId::of::<B>(), TypeId::of::<P>()), Box::new(rule));
}

pub fn eval_rule(backend: &dyn Backend, primitive: &dyn Primitive) -> Option<ErasedEval> {
    let rules = EVAL_DISPATCHER.lock().unwrap();
    rules
        .get(&(backend.as_any().type_id(), primitive.as_any().type_id()))
        .cloned()
}

fn _register<B, P>(rules: &mut HashMap<(TypeId, TypeId), ErasedEval>)
where
    B: Backend + 'static + Sync + Send,
    P: Primitive + 'static + Sync + Send,
    Dispatch<B, P>: Eval<B, P> + 'static,
{
    rules.insert(
        (TypeId::of::<B>(), TypeId::of::<P>()),
        Box::new(Dispatch {
            phantom: std::marker::PhantomData::<(B, P)>,
        }),
    );
}
