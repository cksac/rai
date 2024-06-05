use crate::{
    device::Metal, ops, tensor::TensorLike, Backend, Cpu, Cuda, DType, Device, Eval, Tensor, Type,
    BF16, F16, F32, F64, I64, U32, U8,
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

impl<D, T> Eval<D, ops::Full<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(&self, device: &D, op: &ops::Full<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let t = candle_core::Tensor::full(op.val, output.shape(), dev).unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, ops::Normal<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::FloatDType,
{
    fn eval(&self, device: &D, op: &ops::Normal<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let t = candle_core::Tensor::randn(op.mean, op.std, output.shape(), dev);
            let t = t.unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, ops::Random<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::FloatDType,
{
    fn eval(&self, device: &D, op: &ops::Random<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let t = candle_core::Tensor::rand(op.from, op.to, output.shape(), dev);
            let t = t.unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, ops::Arange<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(&self, device: &D, op: &ops::Arange<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let start = op.start;
            let end = op.stop;
            let step = op.step;
            let t = candle_core::Tensor::arange_step::<T::Repr>(start, end, step, dev).unwrap();
            output.set_data(t);
        });
    }
}

impl<D, T> Eval<D, ops::FromArray<T>> for CandleBackend
where
    D: Device + WithDevice,
    T: Type,
    T::Repr: candle_core::WithDType,
{
    fn eval(&self, device: &D, op: &ops::FromArray<T>, _: &[Tensor], output: &Tensor) {
        device.with_device(|dev| {
            let t = candle_core::Tensor::new(op.data.as_slice(), dev)
                .unwrap()
                .reshape(output.shape())
                .unwrap();
            output.set_data(t);
        });
    }
}

impl<D: Device> Eval<D, ops::Add> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Add, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Sub> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Sub, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Mul> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Mul, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Div> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Div, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device + MatMulCheck> Eval<D, ops::MatMul> for CandleBackend {
    fn eval(&self, device: &D, _: &ops::MatMul, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Sin> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Sin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sin().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Cos> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Cos, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.cos().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Negative> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Negative, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.neg().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::ReduceSum> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ReduceSum, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dims = op.dims();
        let t = if op.keep_dim {
            t.sum_keepdim(dims).unwrap()
        } else {
            t.sum(dims).unwrap()
        };
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::ReduceMax> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ReduceMax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dims = op.dims();
        assert!(
            dims.len() == 1,
            "Candle only support reduce max with single dim"
        );
        let t = if op.keep_dim {
            t.max_keepdim(dims[0]).unwrap()
        } else {
            t.max(dims[0]).unwrap()
        };
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::ReduceMin> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ReduceMin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dims = op.dims();
        assert!(dims.len() == 1, "only support reduce min with single dim");
        let t = if op.keep_dim {
            t.min_keepdim(dims[0]).unwrap()
        } else {
            t.min(dims[0]).unwrap()
        };
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Square> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Square, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqr().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Sqrt> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Sqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Rsqrt> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Rsqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap().recip().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Transpose> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Transpose, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.transpose(op.dim0, op.dim1).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Reshape> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Reshape, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.reshape(op.shape()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Broadcast> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Broadcast, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.broadcast_as(op.shape()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Permute> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Permute, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.permute(op.dims()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Sign> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Sign, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Abs> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Abs, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.abs().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Exp> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Exp, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.exp().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Log> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Log, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.log().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Log2> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Log2, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 2.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Log10> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Log10, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 10.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Equal> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Equal, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::NotEqual> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::NotEqual, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Greater> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Greater, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::GreaterEqual> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::GreaterEqual, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Less> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Less, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::LessEqual> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::LessEqual, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Maximum> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Maximum, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::Minimum> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Minimum, inputs: &[Tensor], output: &Tensor) {
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

impl<D, T> Eval<D, ops::ToDType<T>> for CandleBackend
where
    D: Device,
    T: Type + Into<candle_core::DType>,
{
    fn eval(&self, _: &D, op: &ops::ToDType<T>, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.to_dtype(op.dtype.into()).unwrap();
        output.set_data(t)
    }
}

impl<D1, D2> Eval<D1, ops::ToDevice<D2>> for CandleBackend
where
    D1: Device,
    D2: Device + Clone + WithDevice,
{
    fn eval(&self, _: &D1, op: &ops::ToDevice<D2>, inputs: &[Tensor], output: &Tensor) {
        op.device.with_device(|dev| {
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

impl<D: Device> Eval<D, ops::Softmax> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Softmax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = softmax(t, op.dim).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::LogSoftmax> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::LogSoftmax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = log_softmax(t, op.dim).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Gather> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Gather, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = &t2.deref().contiguous().unwrap();
        let t = t1.gather(t2, op.dim).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::IndexAdd> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::IndexAdd, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let source = &inputs[1];
        let index = &inputs[2];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = index.get_data::<Data>().unwrap();
        let t3 = source.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t3 = t3.deref();
        let t = t1.index_add(t2, t3, op.dim).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::IndexSelect> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::IndexSelect, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = t1.index_select(t2, op.dim).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::Concatenate> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Concatenate, inputs: &[Tensor], output: &Tensor) {
        let tensors: Vec<_> = inputs
            .iter()
            .map(|t| t.get_data::<Data>().unwrap().clone())
            .collect();
        let t = candle_core::Tensor::cat(tensors.as_slice(), op.dim).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Narrow> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Narrow, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.narrow(op.dim, op.start, op.len).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Where> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Where, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::ArgMax> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ArgMax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dim = op.dim();
        let t = if op.keep_dim {
            t.argmax_keepdim(dim).unwrap()
        } else {
            t.argmax(dim).unwrap()
        };
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::ArgMin> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ArgMin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let dim = op.dim();
        let t = if op.keep_dim {
            t.argmax_keepdim(dim).unwrap()
        } else {
            t.argmax(dim).unwrap()
        };
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Erf> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Erf, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.erf().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::Tanh> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::Tanh, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.tanh().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::PowerFloat> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::PowerFloat, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.powf(op.exponent).unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::ToContiguous> for CandleBackend {
    fn eval(&self, _: &D, _: &ops::ToContiguous, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.contiguous().unwrap();
        output.set_data(t)
    }
}

impl<D: Device> Eval<D, ops::ScatterAdd> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ScatterAdd, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let updates = &inputs[1];
        let indices = &inputs[2];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = indices.get_data::<Data>().unwrap();
        let t3 = updates.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref().contiguous().unwrap();
        let t3 = t3.deref().contiguous().unwrap();
        let t = t1.scatter_add(&t2, &t3, op.dim).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::Conv1d> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Conv1d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = t1
            .conv1d(t2, op.padding, op.stride, op.dilation, 1)
            .unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::Conv2d> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::Conv2d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let padding = op.padding.as_slice();
        let stride = op.stride.as_slice();
        let dilation = op.dilation.as_slice();
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

impl<D: Device> Eval<D, ops::ConvTranspose1d> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ConvTranspose1d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = t1
            .conv_transpose1d(t2, op.padding, op.output_padding, op.stride, op.dilation, 1)
            .unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::ConvTranspose2d> for CandleBackend {
    fn eval(&self, _: &D, op: &ops::ConvTranspose2d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let kernel = &inputs[1];
        let t1 = x.get_data::<Data>().unwrap();
        let t2 = kernel.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let padding = op.padding.as_slice();
        let out_padding = op.out_padding.as_slice();
        let stride = op.stride.as_slice();
        let dilation = op.dilation.as_slice();
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

impl<D: Device> Eval<D, ops::MaxPool1d> for CandleBackend {
    fn eval(&self, _: &D, _p: &ops::MaxPool1d, _inputs: &[Tensor], _output: &Tensor) {
        unimplemented!("Candle max_pool1d is not implemented")
    }
}

impl<D: Device> Eval<D, ops::MaxPool2d> for CandleBackend {
    fn eval(&self, _: &D, p: &ops::MaxPool2d, inputs: &[Tensor], output: &Tensor) {
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

impl<D: Device> Eval<D, ops::AvgPool1d> for CandleBackend {
    fn eval(&self, _: &D, _p: &ops::AvgPool1d, _inputs: &[Tensor], _output: &Tensor) {
        unimplemented!("Candle avg_pool1d is not implemented")
    }
}

impl<D: Device> Eval<D, ops::AvgPool2d> for CandleBackend {
    fn eval(&self, _: &D, p: &ops::AvgPool2d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let t1 = x.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        assert_eq!(p.padding.0, 0, "Candle avg_pool2d only support padding=0");
        assert_eq!(p.padding.1, 0, "Candle avg_pool2d only support padding=0");
        let t = t1.avg_pool2d_with_stride(p.kernel_size, p.stride).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::UpsampleNearest1d> for CandleBackend {
    fn eval(&self, _: &D, p: &ops::UpsampleNearest1d, inputs: &[Tensor], output: &Tensor) {
        let x: &Tensor = &inputs[0];
        let t1 = x.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t = t1.upsample_nearest1d(p.size).unwrap();
        output.set_data(t);
    }
}

impl<D: Device> Eval<D, ops::UpsampleNearest2d> for CandleBackend {
    fn eval(&self, _: &D, p: &ops::UpsampleNearest2d, inputs: &[Tensor], output: &Tensor) {
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
impl Eval<Cuda, ops::FlashAttention> for CandleBackend {
    fn eval(&self, _: &Cuda, op: &ops::FlashAttention, inputs: &[Tensor], output: &Tensor) {
        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        let t1 = q.get_data::<Data>().unwrap();
        let t2 = k.get_data::<Data>().unwrap();
        let t3 = v.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t3 = t3.deref();
        let t = match &op.alibi_slopes {
            Some(alibi_slopes) => {
                let t4 = alibi_slopes.get_data::<Data>().unwrap();
                let t4 = t4.deref();
                candle_flash_attn::flash_attn_alibi_windowed(
                    t1,
                    t2,
                    t3,
                    t4,
                    op.softmax_scale,
                    op.window_size_left,
                    op.window_size_right,
                )
                .unwrap()
            }
            None => candle_flash_attn::flash_attn_windowed(
                t1,
                t2,
                t3,
                op.softmax_scale,
                op.window_size_left,
                op.window_size_right,
            )
            .unwrap(),
        };
        output.set_data(t);
    }
}
