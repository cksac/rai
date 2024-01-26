use crate::{
    primitives, tensor::TensorLike, utils::dot_graph, Backend, Cpu, Cuda, DType, Device, Eval,
    Shape, Tensor, Type, BF16, F16, F32, F64, I64, U32, U8,
};
use candle_core::{backend::BackendDevice, CudaDevice};
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

    fn dtype(&self) -> &dyn DType {
        match self.dtype() {
            candle_core::DType::F32 => &F32,
            candle_core::DType::F64 => &F64,
            candle_core::DType::U8 => &U8,
            candle_core::DType::U32 => &U32,
            candle_core::DType::I64 => &I64,
            candle_core::DType::BF16 => &BF16,
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

impl From<BF16> for candle_core::DType {
    fn from(_: BF16) -> Self {
        candle_core::DType::BF16
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

impl From<I64> for candle_core::DType {
    fn from(_: I64) -> Self {
        candle_core::DType::I64
    }
}

impl From<Cpu> for candle_core::Device {
    fn from(_: Cpu) -> Self {
        candle_core::Device::Cpu
    }
}

impl<'a> From<&'a Cpu> for candle_core::Device {
    fn from(_: &'a Cpu) -> Self {
        candle_core::Device::Cpu
    }
}

impl From<Cuda> for candle_core::Device {
    fn from(device: Cuda) -> Self {
        candle_core::Device::Cuda(CudaDevice::new(device.0).expect("cuda"))
    }
}

impl<'a> From<&'a Cuda> for candle_core::Device {
    fn from(device: &'a Cuda) -> Self {
        candle_core::Device::Cuda(CudaDevice::new(device.0).expect("cuda"))
    }
}

impl<D, T> Eval<D, primitives::Full<T>> for CandleBackend
where
    D: Device,
    for<'a> &'a D: Into<candle_core::Device>,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Full<T>, _: &[Tensor], output: &Tensor) {
        let device = &Into::<candle_core::Device>::into(device);
        let t = candle_core::Tensor::full(primitive.val, output.shape(), device).unwrap();
        output.set_data(t);
    }
}

impl<D, T> Eval<D, primitives::Normal<T>> for CandleBackend
where
    D: Device,
    for<'a> &'a D: Into<candle_core::Device>,
    T: Type,
    T::Repr: candle_core::FloatDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Normal<T>, _: &[Tensor], output: &Tensor) {
        let device = &Into::<candle_core::Device>::into(device);
        let t = candle_core::Tensor::randn(primitive.mean, primitive.std, output.shape(), device);
        let t = t.unwrap();
        output.set_data(t);
    }
}

impl<D, T> Eval<D, primitives::Random<T>> for CandleBackend
where
    D: Device,
    for<'a> &'a D: Into<candle_core::Device>,
    T: Type,
    T::Repr: candle_core::FloatDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Random<T>, _: &[Tensor], output: &Tensor) {
        let device = &Into::<candle_core::Device>::into(device);
        let t = candle_core::Tensor::rand(primitive.from, primitive.to, output.shape(), device);
        let t = t.unwrap();
        output.set_data(t);
    }
}

impl<D, T> Eval<D, primitives::Arange<T>> for CandleBackend
where
    D: Device,
    for<'a> &'a D: Into<candle_core::Device>,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Arange<T>, _: &[Tensor], output: &Tensor) {
        let device = &Into::<candle_core::Device>::into(device);
        let start = primitive.start;
        let end = primitive.stop;
        let step = primitive.step;
        let t = candle_core::Tensor::arange_step::<T::Repr>(start, end, step, device).unwrap();
        output.set_data(t);
    }
}

impl<D, T> Eval<D, primitives::FromArray<T>> for CandleBackend
where
    D: Device,
    for<'a> &'a D: Into<candle_core::Device>,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(
        &self,
        device: &D,
        primitive: &primitives::FromArray<T>,
        _: &[Tensor],
        output: &Tensor,
    ) {
        let device = &Into::<candle_core::Device>::into(device);
        let t = candle_core::Tensor::new(primitive.data.as_slice(), device)
            .unwrap()
            .reshape(output.shape())
            .unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::Add> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Add, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Sub> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Sub, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Mul> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Mul, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Div> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Div, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::MatMul> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::MatMul, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Sin> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Sin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sin().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Cos> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Cos, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.cos().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Negative> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Negative, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.neg().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::ReduceSum> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::ReduceSum, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::ReduceMax> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::ReduceMax, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::ReduceMin> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::ReduceMin, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Square> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Square, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqr().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Sqrt> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Sqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Rsqrt> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Rsqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap().recip().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Transpose> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Transpose, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.transpose(primitive.dim0, primitive.dim1).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Reshape> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Reshape, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.reshape(primitive.shape()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Broadcast> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Broadcast, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.broadcast_as(primitive.shape()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Sign> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Sign, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let zero = t.zeros_like().unwrap();
        let t = (t.ge(&zero).unwrap() - t.le(&zero).unwrap()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Abs> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Abs, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.abs().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Exp> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Exp, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.exp().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Log> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Log, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.log().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Log2> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Log2, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 2.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Log10> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Log10, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 10.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Equal> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Equal, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::NotEqual> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::NotEqual, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Greater> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Greater, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::GreaterEqual> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::GreaterEqual, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Less> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Less, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::LessEqual> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::LessEqual, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Maximum> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Maximum, inputs: &[Tensor], output: &Tensor) {
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

impl<D, T> Eval<D, primitives::ToDType<T>> for CandleBackend
where
    D: Device,
    T: Type + Into<candle_core::DType>,
{
    fn eval(&self, _: &D, primitive: &primitives::ToDType<T>, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.to_dtype(primitive.dtype.into()).unwrap();
        output.set_data(t)
    }
}

impl<D1, D2> Eval<D1, primitives::ToDevice<D2>> for CandleBackend
where
    D1: Device,
    D2: Device + Clone,
    for<'a> &'a D2: Into<candle_core::Device>,
{
    fn eval(
        &self,
        _: &D1,
        primitive: &primitives::ToDevice<D2>,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let device = &Into::<candle_core::Device>::into(&primitive.device);
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.to_device(device).unwrap();
        output.set_data(t)
    }
}

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

impl<D: Device> Eval<D, primitives::Softmax> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Softmax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = softmax(t, primitive.dim).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::LogSoftmax> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::LogSoftmax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = log_softmax(t, primitive.dim).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Gather> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Gather, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = t1.gather(t2, primitive.dim).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::IndexSelect> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::IndexSelect, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Concatenate> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Concatenate, inputs: &[Tensor], output: &Tensor) {
        let tensors: Vec<_> = inputs
            .iter()
            .map(|t| t.get_data::<Data>().unwrap().clone())
            .collect();
        let t = candle_core::Tensor::cat(tensors.as_slice(), primitive.dim).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Narrow> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Narrow, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t
            .narrow(primitive.dim, primitive.start, primitive.len)
            .unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Where> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Where, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::ArgMax> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::ArgMax, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::ArgMin> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::ArgMin, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, primitives::Erf> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Erf, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.erf().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Tanh> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Tanh, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.tanh().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::PowerFloat> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::PowerFloat, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.powf(primitive.exponent).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::ToContiguous> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::ToContiguous, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.contiguous().unwrap();
        output.set_data(t)
    }
}
