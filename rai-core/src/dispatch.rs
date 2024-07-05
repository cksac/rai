use crate::{ops, CandleBackend, Cpu, Device, Eval, Op, Tensor, BF16, F16, F32, F64, I64, U32, U8};
use once_cell::sync::Lazy;
use std::{any::TypeId, collections::HashMap, sync::RwLock};

#[derive(Debug, Clone)]
pub struct BackendWrapper<D, P, B> {
    backend: B,
    phantom: std::marker::PhantomData<fn(D, P)>,
}

impl<D, P, B> Eval<dyn Device, dyn Op> for BackendWrapper<D, P, B>
where
    D: Device + 'static + Clone,
    P: Op + 'static + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    #[inline]
    fn eval(&self, device: &dyn Device, op: &dyn Op, inputs: &[Tensor], output: &Tensor) {
        let device = device.as_any().downcast_ref::<D>().unwrap();
        let op = op.as_any().downcast_ref::<P>().unwrap();
        self.backend.eval(device, op, inputs, output);
    }
}

impl<D, P, B> Eval<Box<dyn Device>, Box<dyn Op>> for BackendWrapper<D, P, B>
where
    D: Device + 'static + Clone,
    P: Op + 'static + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    #[inline]
    fn eval(&self, device: &Box<dyn Device>, op: &Box<dyn Op>, inputs: &[Tensor], output: &Tensor) {
        let device = device.as_any().downcast_ref::<D>().unwrap();
        let op = op.as_any().downcast_ref::<P>().unwrap();
        self.backend.eval(device, op, inputs, output);
    }
}

type DynBackend = Box<dyn Eval<dyn Device, dyn Op>>;

macro_rules! register_backend {
    ($backend:ident, $device:ty, $rules:expr) => {
        // creation
        _register::<$backend, $device, ops::Full<U8>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Full<U32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Full<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Full<F16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Full<F32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Full<F64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Full<I64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Random<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Random<F16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Random<F32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Random<F64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Normal<F16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Normal<F32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Normal<F64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Arange<U8>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Arange<U32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Arange<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Arange<F16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Arange<F32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Arange<F64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Arange<I64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::FromArray<U8>>($backend, &mut $rules);
        _register::<$backend, $device, ops::FromArray<U32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::FromArray<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::FromArray<F16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::FromArray<F32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::FromArray<F64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::FromArray<I64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Concatenate>($backend, &mut $rules);

        // binary
        _register::<$backend, $device, ops::Add>($backend, &mut $rules);
        _register::<$backend, $device, ops::Sub>($backend, &mut $rules);
        _register::<$backend, $device, ops::Mul>($backend, &mut $rules);
        _register::<$backend, $device, ops::Div>($backend, &mut $rules);
        _register::<$backend, $device, ops::MatMul>($backend, &mut $rules);
        _register::<$backend, $device, ops::Equal>($backend, &mut $rules);
        _register::<$backend, $device, ops::NotEqual>($backend, &mut $rules);
        _register::<$backend, $device, ops::Greater>($backend, &mut $rules);
        _register::<$backend, $device, ops::GreaterEqual>($backend, &mut $rules);
        _register::<$backend, $device, ops::Less>($backend, &mut $rules);
        _register::<$backend, $device, ops::LessEqual>($backend, &mut $rules);
        _register::<$backend, $device, ops::Maximum>($backend, &mut $rules);
        _register::<$backend, $device, ops::Minimum>($backend, &mut $rules);

        // unary
        _register::<$backend, $device, ops::Sin>($backend, &mut $rules);
        _register::<$backend, $device, ops::Cos>($backend, &mut $rules);
        _register::<$backend, $device, ops::Tanh>($backend, &mut $rules);
        _register::<$backend, $device, ops::Negative>($backend, &mut $rules);
        _register::<$backend, $device, ops::Square>($backend, &mut $rules);
        _register::<$backend, $device, ops::Powf<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Powf<F16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Powf<F32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Powf<F64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Sqrt>($backend, &mut $rules);
        _register::<$backend, $device, ops::Rsqrt>($backend, &mut $rules);
        _register::<$backend, $device, ops::Sign>($backend, &mut $rules);
        _register::<$backend, $device, ops::Abs>($backend, &mut $rules);
        _register::<$backend, $device, ops::Exp>($backend, &mut $rules);
        _register::<$backend, $device, ops::Log>($backend, &mut $rules);
        _register::<$backend, $device, ops::Log2>($backend, &mut $rules);
        _register::<$backend, $device, ops::Log10>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDType<U8>>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDType<U32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDType<BF16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDType<F16>>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDType<F32>>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDType<F64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDType<I64>>($backend, &mut $rules);
        _register::<$backend, $device, ops::Softmax>($backend, &mut $rules);
        _register::<$backend, $device, ops::LogSoftmax>($backend, &mut $rules);
        _register::<$backend, $device, ops::Erf>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToDevice<Cpu>>($backend, &mut $rules);
        #[cfg(feature = "cuda")]
        _register::<$backend, $device, ops::ToDevice<crate::Cuda>>($backend, &mut $rules);
        #[cfg(feature = "metal")]
        _register::<$backend, $device, ops::ToDevice<crate::Metal>>($backend, &mut $rules);

        // indexing
        _register::<$backend, $device, ops::Gather>($backend, &mut $rules);
        _register::<$backend, $device, ops::IndexSelect>($backend, &mut $rules);
        _register::<$backend, $device, ops::Narrow>($backend, &mut $rules);
        _register::<$backend, $device, ops::Where>($backend, &mut $rules);
        _register::<$backend, $device, ops::ScatterAdd>($backend, &mut $rules);
        _register::<$backend, $device, ops::IndexAdd>($backend, &mut $rules);

        // transform
        _register::<$backend, $device, ops::Transpose>($backend, &mut $rules);
        _register::<$backend, $device, ops::Reshape>($backend, &mut $rules);
        _register::<$backend, $device, ops::Permute>($backend, &mut $rules);
        _register::<$backend, $device, ops::Broadcast>($backend, &mut $rules);
        _register::<$backend, $device, ops::ToContiguous>($backend, &mut $rules);

        // reduce
        _register::<$backend, $device, ops::ReduceSum>($backend, &mut $rules);
        _register::<$backend, $device, ops::ReduceMax>($backend, &mut $rules);
        _register::<$backend, $device, ops::ReduceMin>($backend, &mut $rules);
        _register::<$backend, $device, ops::ArgMax>($backend, &mut $rules);
        _register::<$backend, $device, ops::ArgMin>($backend, &mut $rules);

        // others
        _register::<$backend, $device, ops::Conv1d>($backend, &mut $rules);
        _register::<$backend, $device, ops::Conv2d>($backend, &mut $rules);
        _register::<$backend, $device, ops::ConvTranspose1d>($backend, &mut $rules);
        _register::<$backend, $device, ops::ConvTranspose2d>($backend, &mut $rules);
        _register::<$backend, $device, ops::MaxPool1d>($backend, &mut $rules);
        _register::<$backend, $device, ops::MaxPool2d>($backend, &mut $rules);
        _register::<$backend, $device, ops::AvgPool1d>($backend, &mut $rules);
        _register::<$backend, $device, ops::AvgPool2d>($backend, &mut $rules);
        _register::<$backend, $device, ops::UpsampleNearest1d>($backend, &mut $rules);
        _register::<$backend, $device, ops::UpsampleNearest2d>($backend, &mut $rules);
    };
}

static EVAL_DISPATCHER: Lazy<RwLock<HashMap<(TypeId, TypeId), DynBackend>>> = Lazy::new(|| {
    let mut rules: HashMap<(TypeId, TypeId), DynBackend> = HashMap::new();

    #[cfg(feature = "candle-backend")]
    register_backend!(CandleBackend, Cpu, rules);

    #[cfg(all(feature = "candle-backend", feature = "cuda"))]
    register_backend!(CandleBackend, crate::Cuda, rules);

    #[cfg(all(feature = "candle-backend", feature = "metal"))]
    register_backend!(CandleBackend, crate::Metal, rules);

    #[cfg(all(
        feature = "candle-backend",
        feature = "cuda",
        feature = "candle-flash-attn"
    ))]
    _register::<CandleBackend, crate::Cuda, ops::FlashAttention>(CandleBackend, rules);

    RwLock::new(rules)
});

pub fn register<D, P, B>(backend: B)
where
    D: Device + 'static + Clone,
    P: Op + 'static + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    let mut dispatcher = EVAL_DISPATCHER.write().unwrap();
    _register::<B, D, P>(backend, &mut dispatcher);
}

fn _register<B, D, P>(backend: B, dispatcher: &mut HashMap<(TypeId, TypeId), DynBackend>)
where
    D: Device + 'static + Clone,
    P: Op + 'static + Clone,
    B: Eval<D, P> + 'static + Clone,
{
    dispatcher.insert(
        (TypeId::of::<D>(), TypeId::of::<P>()),
        Box::new(BackendWrapper {
            backend,
            phantom: std::marker::PhantomData::<fn(D, P)>,
        }),
    );
}

#[inline(always)]
pub fn eval_rule(device: &dyn Device, op: &dyn Op) -> Option<DynBackend> {
    let dispatcher = EVAL_DISPATCHER.read().unwrap();
    dispatcher
        .get(&(device.as_any().type_id(), op.as_any().type_id()))
        .cloned()
}
