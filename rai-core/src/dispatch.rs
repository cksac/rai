use std::{any::TypeId, collections::HashMap, sync::Mutex};

use dyn_clone::DynClone;
use once_cell::sync::Lazy;

use crate::{backend::Cpu, primitives, Backend, Primitive, Tensor, F16, F32, F64, U32, U8};

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

macro_rules! register_backend {
    ($backend:ident, $rules:expr) => {
        // creation
        _register::<$backend, primitives::Full<U8>>(&mut $rules);
        _register::<$backend, primitives::Full<U32>>(&mut $rules);
        _register::<$backend, primitives::Full<F16>>(&mut $rules);
        _register::<$backend, primitives::Full<F32>>(&mut $rules);
        _register::<$backend, primitives::Full<F64>>(&mut $rules);
        _register::<$backend, primitives::Normal>(&mut $rules);
        _register::<$backend, primitives::Arange<U8>>(&mut $rules);
        _register::<$backend, primitives::Arange<U32>>(&mut $rules);
        _register::<$backend, primitives::Arange<F16>>(&mut $rules);
        _register::<$backend, primitives::Arange<F32>>(&mut $rules);
        _register::<$backend, primitives::Arange<F64>>(&mut $rules);
        _register::<$backend, primitives::FromArray<U8>>(&mut $rules);
        _register::<$backend, primitives::FromArray<U32>>(&mut $rules);
        _register::<$backend, primitives::FromArray<F16>>(&mut $rules);
        _register::<$backend, primitives::FromArray<F32>>(&mut $rules);
        _register::<$backend, primitives::FromArray<F64>>(&mut $rules);
        _register::<$backend, primitives::Concatenate>(&mut $rules);

        // binary
        _register::<$backend, primitives::Add>(&mut $rules);
        _register::<$backend, primitives::Sub>(&mut $rules);
        _register::<$backend, primitives::Mul>(&mut $rules);
        _register::<$backend, primitives::Div>(&mut $rules);
        _register::<$backend, primitives::MatMul>(&mut $rules);
        _register::<$backend, primitives::Equal>(&mut $rules);
        _register::<$backend, primitives::NotEqual>(&mut $rules);
        _register::<$backend, primitives::Greater>(&mut $rules);
        _register::<$backend, primitives::GreaterEqual>(&mut $rules);
        _register::<$backend, primitives::Less>(&mut $rules);
        _register::<$backend, primitives::LessEqual>(&mut $rules);
        _register::<$backend, primitives::Maximum>(&mut $rules);

        // unary
        _register::<$backend, primitives::Sin>(&mut $rules);
        _register::<$backend, primitives::Cos>(&mut $rules);
        _register::<$backend, primitives::Tanh>(&mut $rules);
        _register::<$backend, primitives::Negative>(&mut $rules);
        _register::<$backend, primitives::Square>(&mut $rules);
        _register::<$backend, primitives::PowerFloat>(&mut $rules);
        _register::<$backend, primitives::Sqrt>(&mut $rules);
        _register::<$backend, primitives::Rsqrt>(&mut $rules);
        _register::<$backend, primitives::Sign>(&mut $rules);
        _register::<$backend, primitives::Abs>(&mut $rules);
        _register::<$backend, primitives::Exp>(&mut $rules);
        _register::<$backend, primitives::Log>(&mut $rules);
        _register::<$backend, primitives::Log2>(&mut $rules);
        _register::<$backend, primitives::Log10>(&mut $rules);
        _register::<$backend, primitives::AsType<U8>>(&mut $rules);
        _register::<$backend, primitives::AsType<U32>>(&mut $rules);
        _register::<$backend, primitives::AsType<F16>>(&mut $rules);
        _register::<$backend, primitives::AsType<F32>>(&mut $rules);
        _register::<$backend, primitives::AsType<F64>>(&mut $rules);
        _register::<$backend, primitives::Softmax>(&mut $rules);
        _register::<$backend, primitives::LogSoftmax>(&mut $rules);
        _register::<$backend, primitives::Erf>(&mut $rules);

        // indexing
        _register::<$backend, primitives::Gather>(&mut $rules);
        _register::<$backend, primitives::IndexSelect>(&mut $rules);
        _register::<$backend, primitives::Narrow>(&mut $rules);
        _register::<$backend, primitives::Where>(&mut $rules);

        // transform
        _register::<$backend, primitives::Transpose>(&mut $rules);
        _register::<$backend, primitives::Reshape>(&mut $rules);
        _register::<$backend, primitives::Broadcast>(&mut $rules);
        _register::<$backend, primitives::ToContiguous>(&mut $rules);

        // reduce
        _register::<$backend, primitives::ReduceSum>(&mut $rules);
        _register::<$backend, primitives::ReduceMax>(&mut $rules);
        _register::<$backend, primitives::ReduceMin>(&mut $rules);
        _register::<$backend, primitives::ArgMax>(&mut $rules);
        _register::<$backend, primitives::ArgMin>(&mut $rules);
    };
}

static EVAL_DISPATCHER: Lazy<Mutex<HashMap<(TypeId, TypeId), ErasedEval>>> = Lazy::new(|| {
    let mut rules: HashMap<(TypeId, TypeId), ErasedEval> = HashMap::new();

    register_backend!(Cpu, rules);

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
