use crate::{
    primitives, tensor::TensorLike, utils::dot_graph, Backend, Cpu, DType, Device, DynDType, Eval,
    Shape, Tensor, F16, F32, F64, U32, U8,
};
use half::{bf16, f16};
use safetensors::View;
use std::{
    any::{Any, TypeId},
    ops::Deref,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CandleBackend;

type Data = candle_core::Tensor;

impl TensorLike for candle_core::Tensor {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.dims()
    }

    fn dtype(&self) -> &dyn DynDType {
        match self.dtype() {
            candle_core::DType::F32 => &F32,
            candle_core::DType::F64 => &F64,
            candle_core::DType::U8 => &U8,
            candle_core::DType::U32 => &U32,
            candle_core::DType::I64 => todo!(),
            candle_core::DType::BF16 => todo!(),
            candle_core::DType::F16 => &F16,
        }
    }

    fn as_scalar(&self) -> Box<dyn std::any::Any> {
        match self.dtype() {
            candle_core::DType::F32 => Box::new(self.to_scalar::<f32>().unwrap()),
            candle_core::DType::F64 => Box::new(self.to_scalar::<f64>().unwrap()),
            candle_core::DType::U8 => Box::new(self.to_scalar::<u8>().unwrap()),
            candle_core::DType::U32 => Box::new(self.to_scalar::<u32>().unwrap()),
            candle_core::DType::I64 => Box::new(self.to_scalar::<i64>().unwrap()),
            candle_core::DType::BF16 => Box::new(self.to_scalar::<bf16>().unwrap()),
            candle_core::DType::F16 => Box::new(self.to_scalar::<f16>().unwrap()),
        }
    }

    fn as_vec(&self) -> Box<dyn Any> {
        match self.dtype() {
            candle_core::DType::F32 => Box::new(self.to_vec1::<f32>().unwrap()),
            candle_core::DType::F64 => Box::new(self.to_vec1::<f64>().unwrap()),
            candle_core::DType::U8 => Box::new(self.to_vec1::<u8>().unwrap()),
            candle_core::DType::U32 => Box::new(self.to_vec1::<u32>().unwrap()),
            candle_core::DType::I64 => Box::new(self.to_vec1::<i64>().unwrap()),
            candle_core::DType::BF16 => Box::new(self.to_vec1::<bf16>().unwrap()),
            candle_core::DType::F16 => Box::new(self.to_vec1::<f16>().unwrap()),
        }
    }

    fn as_bytes(&self) -> Vec<u8> {
        View::data(self).into_owned()
    }
}

impl Backend for CandleBackend {
    fn clone_boxed(&self) -> Box<dyn Backend> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn data_type_id(&self) -> std::any::TypeId {
        TypeId::of::<Data>()
    }

    fn equal(&self, rhs: &dyn Backend) -> bool {
        rhs.as_any()
            .downcast_ref()
            .map_or(false, |other| self == other)
    }
}

impl From<U8> for candle_core::DType {
    fn from(_: U8) -> Self {
        candle_core::DType::U8
    }
}

impl From<U32> for candle_core::DType {
    fn from(_: U32) -> Self {
        candle_core::DType::U32
    }
}

impl From<F16> for candle_core::DType {
    fn from(_: F16) -> Self {
        candle_core::DType::F16
    }
}

impl From<F32> for candle_core::DType {
    fn from(_: F32) -> Self {
        candle_core::DType::F32
    }
}

impl From<F64> for candle_core::DType {
    fn from(_: F64) -> Self {
        candle_core::DType::F64
    }
}

impl From<&Box<dyn Device>> for candle_core::Device {
    fn from(val: &Box<dyn Device>) -> Self {
        let d = val.as_any();
        if d.downcast_ref::<Cpu>().is_some() {
            return candle_core::Device::Cpu;
        }
        panic!("unsupported device: {:?}", val);
    }
}

impl From<&dyn Device> for candle_core::Device {
    fn from(val: &dyn Device) -> Self {
        let d = val.as_any();
        if d.downcast_ref::<Cpu>().is_some() {
            return candle_core::Device::Cpu;
        }
        panic!("unsupported device: {:?}", val);
    }
}

macro_rules! impl_full {
    ($T:ty) => {
        impl Eval<Cpu, primitives::Full<$T>> for CandleBackend {
            fn eval(
                &self,
                _: &Cpu,
                primitive: &primitives::Full<$T>,
                _: &[Tensor],
                output: &Tensor,
            ) {
                let t = candle_core::Tensor::full(
                    primitive.val,
                    output.shape(),
                    &output.device().into(),
                )
                .unwrap();
                output.set_data(t);
            }
        }
    };
}

impl_full!(U8);
impl_full!(U32);
impl_full!(F16);
impl_full!(F32);
impl_full!(F64);

impl Eval<Cpu, primitives::Normal> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Normal, _: &[Tensor], output: &Tensor) {
        // TODO: not always f32
        let t = candle_core::Tensor::rand(-1.0f32, 1.0f32, output.shape(), &output.device().into());
        let t = t.unwrap();
        output.set_data(t);
    }
}

macro_rules! impl_arange {
    ($T:ty) => {
        impl Eval<Cpu, primitives::Arange<$T>> for CandleBackend {
            fn eval(
                &self,
                _: &Cpu,
                primitive: &primitives::Arange<$T>,
                _: &[Tensor],
                output: &Tensor,
            ) {
                let start = primitive.start;
                let end = primitive.stop;
                let step = primitive.step;
                let t = candle_core::Tensor::arange_step::<<$T as DType>::Repr>(
                    start,
                    end,
                    step,
                    &output.device().into(),
                )
                .unwrap();
                output.set_data(t);
            }
        }
    };
}

impl_arange!(U8);
impl_arange!(U32);
impl_arange!(F16);
impl_arange!(F32);
impl_arange!(F64);

macro_rules! impl_from_array {
    ($T:ty) => {
        impl Eval<Cpu, primitives::FromArray<$T>> for CandleBackend {
            fn eval(
                &self,
                _: &Cpu,
                primitive: &primitives::FromArray<$T>,
                _: &[Tensor],
                output: &Tensor,
            ) {
                let device = &output.device().into();
                let t = candle_core::Tensor::new(primitive.data.as_slice(), device)
                    .unwrap()
                    .reshape(output.shape())
                    .unwrap();
                output.set_data(t);
            }
        }
    };
}

impl_from_array!(U8);
impl_from_array!(U32);
impl_from_array!(F16);
impl_from_array!(F32);
impl_from_array!(F64);

impl Eval<Cpu, primitives::Add> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Add, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            (t1 + t2).unwrap()
        } else {
            t1.broadcast_add(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Sub> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Sub, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            (t1 - t2).unwrap()
        } else {
            t1.broadcast_sub(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Mul> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Mul, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            (t1 * t2).unwrap_or_else(|e| {
                panic!(
                    "Mul({:?}, {:?}) with error {:?}\n{}",
                    lhs,
                    rhs,
                    e,
                    dot_graph([lhs, rhs])
                )
            })
        } else {
            t1.broadcast_mul(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Div> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Div, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            (t1 / t2).unwrap()
        } else {
            t1.broadcast_div(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::MatMul> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::MatMul, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        // todo: get is required broadcast info in primitives::MatMul
        let lhs_sp = lhs.shape_of(..-2);
        let rhs_sp = rhs.shape_of(..-2);
        let t = if lhs_sp != rhs_sp {
            t1.broadcast_matmul(t2).unwrap()
        } else {
            t1.matmul(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Sin> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Sin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sin().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Cos> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Cos, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.cos().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Negative> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Negative, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.neg().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::ReduceSum> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::ReduceSum, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dims = primitive.dims();
        let t = if primitive.keep_dim {
            t.sum_keepdim(dims).unwrap()
        } else {
            t.sum(dims).unwrap()
        };
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::ReduceMax> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::ReduceMax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dims = primitive.dims();
        assert!(dims.len() == 1, "only support reduce max with single dim");
        let t = if primitive.keep_dim {
            t.max_keepdim(dims[0]).unwrap()
        } else {
            t.max(dims[0]).unwrap()
        };
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::ReduceMin> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::ReduceMin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dims = primitive.dims();
        assert!(dims.len() == 1, "only support reduce min with single dim");
        let t = if primitive.keep_dim {
            t.min_keepdim(dims[0]).unwrap()
        } else {
            t.min(dims[0]).unwrap()
        };
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Square> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Square, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqr().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Sqrt> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Sqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Rsqrt> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Rsqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap().recip().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Transpose> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::Transpose, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.transpose(primitive.dim0, primitive.dim1).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Reshape> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::Reshape, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.reshape(primitive.shape()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Broadcast> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::Broadcast, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.broadcast_as(primitive.shape()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Sign> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Sign, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let zero = t.zeros_like().unwrap();
        let t = (t.ge(&zero).unwrap() - t.le(&zero).unwrap()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Abs> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Abs, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.abs().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Exp> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Exp, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.exp().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Log> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Log, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.log().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Log2> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Log2, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 2.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Log10> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Log10, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 10.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Equal> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Equal, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            t1.eq(t2).unwrap()
        } else {
            t1.broadcast_eq(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::NotEqual> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::NotEqual, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            t1.ne(t2).unwrap()
        } else {
            t1.broadcast_ne(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Greater> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Greater, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            t1.gt(t2).unwrap()
        } else {
            t1.broadcast_gt(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::GreaterEqual> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::GreaterEqual, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            t1.ge(t2).unwrap()
        } else {
            t1.broadcast_ge(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Less> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Less, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            t1.lt(t2).unwrap()
        } else {
            t1.broadcast_lt(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::LessEqual> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::LessEqual, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            t1.le(t2).unwrap()
        } else {
            t1.broadcast_le(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Maximum> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Maximum, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if lhs.shape_eq(rhs) {
            t1.maximum(t2).unwrap()
        } else {
            t1.broadcast_maximum(t2).unwrap()
        };
        output.set_data(t);
    }
}

macro_rules! impl_as_type {
    ($T:ty) => {
        impl Eval<Cpu, primitives::AsType<$T>> for CandleBackend {
            fn eval(
                &self,
                _: &Cpu,
                primitive: &primitives::AsType<$T>,
                inputs: &[Tensor],
                output: &Tensor,
            ) {
                let x = &inputs[0];
                let t = x.get_data::<Data>().unwrap();
                let t = t.deref();
                let t = t.to_dtype(primitive.dtype.into()).unwrap();
                output.set_data(t)
            }
        }
    };
}
impl_as_type!(U8);
impl_as_type!(U32);
impl_as_type!(F16);
impl_as_type!(F32);
impl_as_type!(F64);

// from candle_nn::ops
fn softmax<D: candle_core::shape::Dim>(
    xs: &candle_core::Tensor,
    dim: D,
) -> candle_core::Result<candle_core::Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

// from candle_nn::ops
fn log_softmax<D: candle_core::shape::Dim>(
    xs: &candle_core::Tensor,
    d: D,
) -> candle_core::Result<candle_core::Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

impl Eval<Cpu, primitives::Softmax> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::Softmax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = softmax(t, primitive.dim).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::LogSoftmax> for CandleBackend {
    fn eval(
        &self,
        _: &Cpu,
        primitive: &primitives::LogSoftmax,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = log_softmax(t, primitive.dim).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Gather> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::Gather, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        // TODO: slice_sizes not used
        let t = t1.gather(t2, primitive.dim).unwrap();
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::IndexSelect> for CandleBackend {
    fn eval(
        &self,
        _: &Cpu,
        primitive: &primitives::IndexSelect,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        // TODO: slice_sizes not used
        let t = t1.index_select(t2, primitive.dim).unwrap();
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Concatenate> for CandleBackend {
    fn eval(
        &self,
        _: &Cpu,
        primitive: &primitives::Concatenate,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let tensors: Vec<_> = inputs
            .iter()
            .map(|t| t.get_data::<Data>().unwrap().clone())
            .collect();
        let t = candle_core::Tensor::cat(tensors.as_slice(), primitive.dim).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Narrow> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::Narrow, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t
            .narrow(primitive.dim, primitive.start, primitive.len)
            .unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Where> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Where, inputs: &[Tensor], output: &Tensor) {
        let cond = &inputs[0];
        let on_true = &inputs[1];
        let on_false = &inputs[2];
        let t1 = cond.get_data::<Data>().unwrap();
        let t2 = on_true.get_data::<Data>().unwrap();
        let t3 = on_false.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t3 = t3.deref();
        let t = t1.where_cond(t2, t3).unwrap();
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::ArgMax> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::ArgMax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dim = primitive.dim();
        let t = if primitive.keep_dim {
            t.argmax_keepdim(dim).unwrap()
        } else {
            t.argmax(dim).unwrap()
        };
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::ArgMin> for CandleBackend {
    fn eval(&self, _: &Cpu, primitive: &primitives::ArgMin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dim = primitive.dim();
        let t = if primitive.keep_dim {
            t.argmax_keepdim(dim).unwrap()
        } else {
            t.argmax(dim).unwrap()
        };
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Erf> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Erf, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.erf().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Tanh> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::Tanh, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.tanh().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::PowerFloat> for CandleBackend {
    fn eval(
        &self,
        _: &Cpu,
        primitive: &primitives::PowerFloat,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.powf(primitive.exponent).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::ToContiguous> for CandleBackend {
    fn eval(&self, _: &Cpu, _: &primitives::ToContiguous, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.contiguous().unwrap();
        output.set_data(t)
    }
}