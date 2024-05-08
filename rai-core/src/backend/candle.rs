use crate::{
    device::Metal, primitives, tensor::TensorLike, Backend, Cpu, Cuda, DType, Device, Eval, Shape,
    Tensor, Type, BF16, F16, F32, F64, I64, U32, U8,
};
use half::{bf16, f16};
use once_cell::sync::Lazy;
use safetensors::View;
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    ops::Deref,
    sync::RwLock,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
        Box::new(*self)
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

trait WithDevice {
    fn with_device<F>(&self, func: F)
    where
        for<'a> F: FnOnce(&'a candle_core::Device);
}

static CPU_DEV: candle_core::Device = candle_core::Device::Cpu;

static CUDA_DEVICES: Lazy<RwLock<HashMap<usize, candle_core::Device>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

static METAL_DEVICES: Lazy<RwLock<HashMap<usize, candle_core::Device>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

impl WithDevice for Cpu {
    #[inline(always)]
    fn with_device<F>(&self, func: F)
    where
        for<'a> F: FnOnce(&'a candle_core::Device),
    {
        func(&CPU_DEV);
    }
}

impl WithDevice for Cuda {
    #[inline(always)]
    fn with_device<F>(&self, func: F)
    where
        for<'a> F: FnOnce(&'a candle_core::Device),
    {
        let devices = CUDA_DEVICES.read().unwrap();
        if let Some(dev) = devices.get(&self.id()) {
            func(dev);
        } else {
            drop(devices);
            let mut devices = CUDA_DEVICES.write().unwrap();
            let dev = devices
                .entry(self.id())
                .or_insert_with(|| candle_core::Device::new_cuda(self.id()).expect("cuda device"));
            func(dev);
        }
    }
}

impl WithDevice for Metal {
    #[inline(always)]
    fn with_device<F>(&self, func: F)
    where
        for<'a> F: FnOnce(&'a candle_core::Device),
    {
        let devices = METAL_DEVICES.read().unwrap();
        if let Some(dev) = devices.get(&self.id()) {
            func(dev);
        } else {
            drop(devices);
            let mut devices = METAL_DEVICES.write().unwrap();
            let dev = devices.entry(self.id()).or_insert_with(|| {
                candle_core::Device::new_metal(self.id()).expect("metal device")
            });
            func(dev);
        }
    }
}

impl<D, T> Eval<D, primitives::Full<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Full<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let t = candle_core::Tensor::full(primitive.val, output.shape(), dev).unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, primitives::Normal<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::FloatDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Normal<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let t = candle_core::Tensor::randn(primitive.mean, primitive.std, output.shape(), dev);
            let t = t.unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, primitives::Random<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::FloatDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Random<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let t = candle_core::Tensor::rand(primitive.from, primitive.to, output.shape(), dev);
            let t = t.unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, primitives::Arange<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(&self, device: &D, primitive: &primitives::Arange<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let start = primitive.start;
            let end = primitive.stop;
            let step = primitive.step;
            let t = candle_core::Tensor::arange_step::<T::Repr>(start, end, step, dev).unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, primitives::FromArray<T>> for CandleBackend
where
    D: Device + WithDevice,
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
        device.with_device(|dev| {
            let t = candle_core::Tensor::new(primitive.data.as_slice(), dev)
                .unwrap()
                .reshape(output.shape())
                .unwrap();
            output.set_data(t);
        });
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
        let t = (t1 + t2).unwrap();
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
        let t = (t1 - t2).unwrap();
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
        let t = (t1 * t2).unwrap();
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
        let t = (t1 / t2).unwrap();
        output.set_data(t);
    }
}

trait MatMulCheck {
    fn check_matmul(
        &self,
        lhs: &[usize],
        rhs: &[usize],
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> (bool, bool);
}

impl MatMulCheck for Cpu {
    // TODO: update check logic for mkl, accelerate by reference candle
    fn check_matmul(
        &self,
        lhs: &[usize],
        rhs: &[usize],
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> (bool, bool) {
        let m = lhs[lhs.len() - 2];
        let k = lhs[lhs.len() - 1];
        let n = rhs[rhs.len() - 1];
        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
        let l_stride_ok = (lhs_m1 == 1 && lhs_m2 == k) || (lhs_m1 == m && lhs_m2 == 1);
        let r_stride_ok = (rhs_m1 == 1 && rhs_m2 == n) || (rhs_m1 == k && rhs_m2 == 1);
        (l_stride_ok, r_stride_ok)
    }
}

impl MatMulCheck for Cuda {
    fn check_matmul(
        &self,
        lhs: &[usize],
        rhs: &[usize],
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> (bool, bool) {
        let m = lhs[lhs.len() - 2];
        let k = lhs[lhs.len() - 1];
        let n = rhs[rhs.len() - 1];
        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
        let l_stride_ok = (lhs_m1 == 1 && lhs_m2 == k) || (lhs_m1 == m && lhs_m2 == 1);
        let r_stride_ok = (rhs_m1 == 1 && rhs_m2 == n) || (rhs_m1 == k && rhs_m2 == 1);
        (l_stride_ok, r_stride_ok)
    }
}

impl MatMulCheck for Metal {
    fn check_matmul(
        &self,
        lhs: &[usize],
        rhs: &[usize],
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> (bool, bool) {
        let m = lhs[lhs.len() - 2];
        let k = lhs[lhs.len() - 1];
        let n = rhs[rhs.len() - 1];
        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
        let l_stride_ok = (lhs_m1 == 1 && lhs_m2 == k) || (lhs_m1 == m && lhs_m2 == 1);
        let r_stride_ok = (rhs_m1 == 1 && rhs_m2 == n) || (rhs_m1 == k && rhs_m2 == 1);
        (l_stride_ok, r_stride_ok)
    }
}

impl<D: Device + MatMulCheck> Eval<D, primitives::MatMul> for CandleBackend {
    fn eval(&self, device: &D, _: &primitives::MatMul, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1_ref = t1.deref();
        let t2_ref = t2.deref();
        // candle only allow matmul with compatible tensors
        let (l_stride_ok, r_stride_ok) = device.check_matmul(
            t1_ref.shape().dims(),
            t2_ref.shape().dims(),
            t1_ref.stride(),
            t2_ref.stride(),
        );
        // set a contiguous tensor to lhs and rhs?
        let t = match (l_stride_ok, r_stride_ok) {
            (true, true) => t1_ref.matmul(t2_ref).unwrap(),
            (true, false) => {
                let t2_c = t2_ref.contiguous().unwrap();
                drop(t2);
                rhs.set_data(t2_c.clone());
                t1_ref.matmul(&t2_c).unwrap()
            }
            (false, true) => {
                let t1_c = t1_ref.contiguous().unwrap();
                drop(t1);
                lhs.set_data(t1_c.clone());
                t1_c.matmul(t2_ref).unwrap()
            }
            (false, false) => {
                let t1_c = t1_ref.contiguous().unwrap();
                let t2_c = t2_ref.contiguous().unwrap();
                drop(t1);
                drop(t2);
                lhs.set_data(t1_c.clone());
                rhs.set_data(t2_c.clone());
                t1_c.matmul(&t2_c).unwrap()
            }
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
        assert!(
            dims.len() == 1,
            "Candle only support reduce max with single dim"
        );
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

impl<D: Device> Eval<D, primitives::Permute> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Permute, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.permute(primitive.dims()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, primitives::Sign> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Sign, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let zero = t.zeros_like().unwrap();
        let tge = t.ge(&zero).unwrap().to_dtype(t.dtype()).unwrap();
        let tle = t.le(&zero).unwrap().to_dtype(t.dtype()).unwrap();
        let t = (tge - tle).unwrap();
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
        let t = t1.eq(t2).unwrap();
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
        let t = t1.ne(t2).unwrap();
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
        let t = t1.gt(t2).unwrap();
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
        let t = t1.ge(t2).unwrap();
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
        let t = t1.lt(t2).unwrap();
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
        let t = t1.le(t2).unwrap();
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
        let t = t1.maximum(t2).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::Minimum> for CandleBackend {
    fn eval(&self, _: &D, _: &primitives::Minimum, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = t1.minimum(t2).unwrap();
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
    D2: Device + Clone + WithDevice,
{
    fn eval(
        &self,
        _: &D1,
        primitive: &primitives::ToDevice<D2>,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        primitive.device.with_device(|dev| {
            let x = &inputs[0];
            let t = x.get_data::<Data>().unwrap();
            let t = t.deref();
            let t = t.to_device(dev).unwrap();
            output.set_data(t)
        });
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
        let t2 = &t2.deref().contiguous().unwrap();
        let t = t1.gather(t2, primitive.dim).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::IndexAdd> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::IndexAdd, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let source = &inputs[1];
        let index = &inputs[2];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = index.get_data::<Data>().unwrap();
        let t3 = source.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t3 = t3.deref();
        let t = t1.index_add(t2, t3, primitive.dim).unwrap();
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
        let on_true = &inputs[0];
        let on_false = &inputs[1];
        let cond = &inputs[2];
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

impl<D: Device> Eval<D, primitives::ScatterAdd> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::ScatterAdd, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let updates = &inputs[1];
        let indices = &inputs[2];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = indices.get_data::<Data>().unwrap();
        let t3 = updates.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref().contiguous().unwrap();
        let t3 = t3.deref().contiguous().unwrap();
        let t = t1.scatter_add(&t2, &t3, primitive.dim).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::Conv1d> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Conv1d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = t1
            .conv1d(
                t2,
                primitive.padding,
                primitive.stride,
                primitive.dilation,
                1,
            )
            .unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::Conv2d> for CandleBackend {
    fn eval(&self, _: &D, primitive: &primitives::Conv2d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let padding = primitive.padding.as_slice();
        let stride = primitive.stride.as_slice();
        let dilation = primitive.dilation.as_slice();
        assert_eq!(padding[0], padding[1], "Candle only support square padding");
        assert_eq!(stride[0], stride[1], "Candle only support square stride");
        assert_eq!(
            dilation[0], dilation[1],
            "Candle only support square dilation"
        );
        let t = t1
            .conv2d(t2, padding[0], stride[0], dilation[0], 1)
            .unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::ConvTranspose1d> for CandleBackend {
    fn eval(
        &self,
        _: &D,
        primitive: &primitives::ConvTranspose1d,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = t1
            .conv_transpose1d(
                t2,
                primitive.padding,
                primitive.output_padding,
                primitive.stride,
                primitive.dilation,
                1,
            )
            .unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::ConvTranspose2d> for CandleBackend {
    fn eval(
        &self,
        _: &D,
        primitive: &primitives::ConvTranspose2d,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let padding = primitive.padding.as_slice();
        let out_padding = primitive.out_padding.as_slice();
        let stride = primitive.stride.as_slice();
        let dilation = primitive.dilation.as_slice();
        assert_eq!(padding[0], padding[1], "Candle only support square padding");
        assert_eq!(
            out_padding[0], out_padding[1],
            "Candle only support square out_padding"
        );
        assert_eq!(stride[0], stride[1], "Candle only support square stride");
        assert_eq!(
            dilation[0], dilation[1],
            "Candle only support square dilation"
        );
        let t = t1
            .conv_transpose2d(t2, padding[0], out_padding[0], stride[0], dilation[0])
            .unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::MaxPool1d> for CandleBackend {
    fn eval(&self, _: &D, _p: &primitives::MaxPool1d, _inputs: &[Tensor], _output: &Tensor) {
        unimplemented!("Candle max_pool1d is not implemented")
    }
}

impl<D: Device> Eval<D, primitives::MaxPool2d> for CandleBackend {
    fn eval(&self, _: &D, p: &primitives::MaxPool2d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let t1 = x.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        assert_eq!(p.padding.0, 0, "Candle max_pool2d only support padding=0");
        assert_eq!(p.padding.1, 0, "Candle max_pool2d only support padding=0");
        assert_eq!(p.dilation.0, 1, "Candle max_pool2d only support dilation=1");
        assert_eq!(p.dilation.1, 1, "Candle max_pool2d only support dilation=1");
        let t = t1.max_pool2d_with_stride(p.kernel_size, p.stride).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::AvgPool1d> for CandleBackend {
    fn eval(&self, _: &D, _p: &primitives::AvgPool1d, _inputs: &[Tensor], _output: &Tensor) {
        unimplemented!("Candle avg_pool1d is not implemented")
    }
}

impl<D: Device> Eval<D, primitives::AvgPool2d> for CandleBackend {
    fn eval(&self, _: &D, p: &primitives::AvgPool2d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let t1 = x.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        assert_eq!(p.padding.0, 0, "Candle avg_pool2d only support padding=0");
        assert_eq!(p.padding.1, 0, "Candle avg_pool2d only support padding=0");
        let t = t1.avg_pool2d_with_stride(p.kernel_size, p.stride).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::UpsampleNearest1d> for CandleBackend {
    fn eval(&self, _: &D, p: &primitives::UpsampleNearest1d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let t1 = x.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t = t1.upsample_nearest1d(p.size).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, primitives::UpsampleNearest2d> for CandleBackend {
    fn eval(&self, _: &D, p: &primitives::UpsampleNearest2d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let t1 = x.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t = t1.upsample_nearest2d(p.size.0, p.size.1).unwrap();
        output.set_data(t);
    }
}

#[cfg(all(
    feature = "candle-backend",
    feature = "cuda",
    feature = "candle-flash-attn"
))]
impl Eval<Cuda, primitives::FlashAttention> for CandleBackend {
    fn eval(
        &self,
        _: &Cuda,
        primitive: &primitives::FlashAttention,
        inputs: &[Tensor],
        output: &Tensor,
    ) {
        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        let t1 = q.get_data::<Data>().unwrap();
        let t2 = k.get_data::<Data>().unwrap();
        let t3 = v.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t3 = t3.deref();
        let t = match &primitive.alibi_slopes {
            Some(alibi_slopes) => {
                let t4 = alibi_slopes.get_data::<Data>().unwrap();
                let t4 = t4.deref();
                candle_flash_attn::flash_attn_alibi_windowed(
                    t1,
                    t2,
                    t3,
                    t4,
                    primitive.softmax_scale,
                    primitive.window_size_left,
                    primitive.window_size_right,
                )
                .unwrap()
            }
            None => candle_flash_attn::flash_attn_windowed(
                t1,
                t2,
                t3,
                primitive.softmax_scale,
                primitive.window_size_left,
                primitive.window_size_right,
            )
            .unwrap(),
        };
        output.set_data(t);
    }
}
