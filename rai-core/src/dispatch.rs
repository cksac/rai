use crate::{
    primitives, CandleBackend, Cpu, Device, Eval, Primitive, Tensor, BF16, F16, F32, F64, I64, U32,
    U8,
};
use once_cell::sync::Lazy;
use std::{any::TypeId, collections::HashMap, sync::Mutex};

#[derive(Debug, Clone)]
pub struct BackendWrapper<D, P, B> {
    backend: B,
    phantom: std::marker::PhantomData<(D, P)>,
}

impl<D, P, B> Eval<dyn Device, dyn Primitive> for BackendWrapper<D, P, B>
where
    D: Device + 'static + Sync + Send + Clone,
    P: Primitive + 'static + Sync + Send + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    #[inline]
    fn eval(
        &self,
        device: &dyn Device,
        primitive: &dyn Primitive,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let device = device.as_any().downcast_ref::<D>().unwrap();
        let primitive = primitive.as_any().downcast_ref::<P>().unwrap();
        self.backend.eval(device, primitive, inputs, output);
    }
}

impl<D, P, B> Eval<Box<dyn Device>, Box<dyn Primitive>> for BackendWrapper<D, P, B>
where
    D: Device + 'static + Sync + Send + Clone,
    P: Primitive + 'static + Sync + Send + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    #[inline]
    fn eval(
        &self,
        device: &Box<dyn Device>,
        primitive: &Box<dyn Primitive>,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let device = device.as_any().downcast_ref::<D>().unwrap();
        let primitive = primitive.as_any().downcast_ref::<P>().unwrap();
        self.backend.eval(device, primitive, inputs, output);
    }
}

type DynBackend = Box<dyn Eval<dyn Device, dyn Primitive>>;

macro_rules! register_backend {
    ($backend:ident, $device:ty, $rules:expr) => {
        // creation
        _register::<$backend, $device, primitives::Full<U8>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Full<U32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Full<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Full<F16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Full<F32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Full<F64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Full<I64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Random<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Random<F16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Random<F32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Random<F64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Normal<F16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Normal<F32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Normal<F64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Arange<U8>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Arange<U32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Arange<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Arange<F16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Arange<F32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Arange<F64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Arange<I64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::FromArray<U8>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::FromArray<U32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::FromArray<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::FromArray<F16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::FromArray<F32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::FromArray<F64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::FromArray<I64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Concatenate>($backend, &mut $rules);

        // binary
        _register::<$backend, $device, primitives::Add>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Sub>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Mul>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Div>($backend, &mut $rules);
        _register::<$backend, $device, primitives::MatMul>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Equal>($backend, &mut $rules);
        _register::<$backend, $device, primitives::NotEqual>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Greater>($backend, &mut $rules);
        _register::<$backend, $device, primitives::GreaterEqual>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Less>($backend, &mut $rules);
        _register::<$backend, $device, primitives::LessEqual>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Maximum>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Minimum>($backend, &mut $rules);

        // unary
        _register::<$backend, $device, primitives::Sin>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Cos>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Tanh>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Negative>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Square>($backend, &mut $rules);
        _register::<$backend, $device, primitives::PowerFloat>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Sqrt>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Rsqrt>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Sign>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Abs>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Exp>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Log>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Log2>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Log10>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDType<U8>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDType<U32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDType<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDType<F16>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDType<F32>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDType<F64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDType<I64>>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Softmax>($backend, &mut $rules);
        _register::<$backend, $device, primitives::LogSoftmax>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Erf>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToDevice<Cpu>>($backend, &mut $rules);
        #[cfg(feature = "cuda")]
        _register::<$backend, $device, primitives::ToDevice<crate::Cuda>>($backend, &mut $rules);

        // indexing
        _register::<$backend, $device, primitives::Gather>($backend, &mut $rules);
        _register::<$backend, $device, primitives::IndexSelect>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Narrow>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Where>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ScatterAdd>($backend, &mut $rules);
        _register::<$backend, $device, primitives::IndexAdd>($backend, &mut $rules);

        // transform
        _register::<$backend, $device, primitives::Transpose>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Reshape>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Permute>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Broadcast>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ToContiguous>($backend, &mut $rules);

        // reduce
        _register::<$backend, $device, primitives::ReduceSum>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ReduceMax>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ReduceMin>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ArgMax>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ArgMin>($backend, &mut $rules);

        // others
        _register::<$backend, $device, primitives::Conv1d>($backend, &mut $rules);
        _register::<$backend, $device, primitives::Conv2d>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ConvTranspose1d>($backend, &mut $rules);
        _register::<$backend, $device, primitives::ConvTranspose2d>($backend, &mut $rules);
        _register::<$backend, $device, primitives::MaxPool1d>($backend, &mut $rules);
        _register::<$backend, $device, primitives::MaxPool2d>($backend, &mut $rules);
    };
}

static EVAL_DISPATCHER: Lazy<Mutex<HashMap<(TypeId, TypeId), DynBackend>>> = Lazy::new(|| {
    let mut rules: HashMap<(TypeId, TypeId), DynBackend> = HashMap::new();

    #[cfg(feature = "candle-backend")]
    register_backend!(CandleBackend, Cpu, rules);

    #[cfg(all(feature = "candle-backend", feature = "cuda"))]
    register_backend!(CandleBackend, crate::Cuda, rules);

    #[cfg(all(
        feature = "candle-backend",
        feature = "cuda",
        feature = "candle-flash-attn"
    ))]
    _register::<CandleBackend, crate::Cuda, primitives::FlashAttention>(CandleBackend, rules);

    Mutex::new(rules)
});

pub fn register<D, P, B>(backend: B)
where
    D: Device + 'static + Sync + Send + Clone,
    P: Primitive + 'static + Sync + Send + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    let mut dispatcher = EVAL_DISPATCHER.lock().unwrap();
    _register::<B, D, P>(backend, &mut dispatcher);
}

fn _register<B, D, P>(backend: B, dispatcher: &mut HashMap<(TypeId, TypeId), DynBackend>)
where
    D: Device + 'static + Sync + Send + Clone,
    P: Primitive + 'static + Sync + Send + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    dispatcher.insert(
        (TypeId::of::<D>(), TypeId::of::<P>()),
        Box::new(BackendWrapper {
            backend,
            phantom: std::marker::PhantomData::<(D, P)>,
        }),
    );
}

pub fn eval_rule(device: &dyn Device, primitive: &dyn Primitive) -> Option<DynBackend> {
    let dispatcher = EVAL_DISPATCHER.lock().unwrap();
    dispatcher
        .get(&(device.as_any().type_id(), primitive.as_any().type_id()))
        .cloned()
}
