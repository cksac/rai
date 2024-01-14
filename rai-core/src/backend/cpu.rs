use std::{
    any::{Any, TypeId},
    ops::Deref,
};

use crate::{
    backend,
    dispatch::{Dispatch, Eval},
    primitives,
    tensor::TensorLike,
    utils::dot_graph,
    Backend, DType, DynDType, Shape, Tensor, F32, F64, U8,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cpu;

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
            candle_core::DType::U32 => todo!(),
            candle_core::DType::I64 => todo!(),
            candle_core::DType::BF16 => todo!(),
            candle_core::DType::F16 => todo!(),
        }
    }
}

impl Backend for Cpu {
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

// impl<'a> From<&'a dyn DType> for candle_core::DType {
//     fn from(val: &'a dyn DType) -> candle_core::DType {
//         todo!()
//         // match val {
//         //     DType::F32 => candle_core::DType::F32,
//         //     DType::F64 => candle_core::DType::F64,
//         //     DType::U8 => candle_core::DType::U8,
//         // }
//     }
// }

impl From<U8> for candle_core::DType {
    fn from(value: U8) -> Self {
        candle_core::DType::U8
    }
}

impl From<F32> for candle_core::DType {
    fn from(value: F32) -> Self {
        candle_core::DType::F32
    }
}

impl From<F64> for candle_core::DType {
    fn from(value: F64) -> Self {
        candle_core::DType::F64
    }
}

impl From<&Box<dyn Backend>> for candle_core::Device {
    fn from(val: &Box<dyn Backend>) -> Self {
        let d = val.as_any();
        if d.downcast_ref::<backend::Cpu>().is_some() {
            return candle_core::Device::Cpu;
        }
        panic!("unsupported backend: {:?}", val);
    }
}

impl From<&dyn Backend> for candle_core::Device {
    fn from(val: &dyn Backend) -> Self {
        let d = val.as_any();
        if d.downcast_ref::<backend::Cpu>().is_some() {
            return candle_core::Device::Cpu;
        }
        panic!("unsupported backend: {:?}", val);
    }
}

macro_rules! impl_full {
    ($T:ty) => {
        impl Eval<Cpu, primitives::Full<$T>> for Dispatch<Cpu, primitives::Full<$T>> {
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
                    &output.backend().into(),
                )
                .unwrap();
                output.set_data(t);
            }
        }
    };
}

impl_full!(U8);
impl_full!(F32);
impl_full!(F64);

impl Eval<Cpu, primitives::Normal> for Dispatch<Cpu, primitives::Normal> {
    fn eval(&self, _: &Cpu, _: &primitives::Normal, _: &[Tensor], output: &Tensor) {
        // TODO: not always f32
        let t =
            candle_core::Tensor::rand(-1.0f32, 1.0f32, output.shape(), &output.backend().into());
        let t = t.unwrap();
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Arange> for Dispatch<Cpu, primitives::Arange> {
    fn eval(&self, _: &Cpu, primitive: &primitives::Arange, _: &[Tensor], output: &Tensor) {
        todo!()
        // let start = primitive.start;
        // let end = primitive.stop;
        // let step = primitive.step;
        // let t = match output.dtype() {
        //     DType::F32 => candle_core::Tensor::arange_step::<f32>(
        //         start as f32,
        //         end as f32,
        //         step as f32,
        //         &output.backend().into(),
        //     ),
        //     DType::F64 => {
        //         candle_core::Tensor::arange_step::<f64>(start, end, step, &output.backend().into())
        //     }
        //     DType::U8 => candle_core::Tensor::arange_step::<u8>(
        //         start as u8,
        //         end as u8,
        //         step as u8,
        //         &output.backend().into(),
        //     ),
        // }
        // .unwrap();
        // output.set_data(t);
    }
}

macro_rules! impl_from_array {
    ($T:ty) => {
        impl Eval<Cpu, primitives::FromArray<$T>> for Dispatch<Cpu, primitives::FromArray<$T>> {
            fn eval(
                &self,
                _: &Cpu,
                primitive: &primitives::FromArray<$T>,
                _: &[Tensor],
                output: &Tensor,
            ) {
                let device = &output.backend().into();
                let t = candle_core::Tensor::new(primitive.data.as_slice(), device).unwrap();
                output.set_data(t);
            }
        }
    };
}

impl_from_array!(U8);
impl_from_array!(F32);
impl_from_array!(F64);

impl Eval<Cpu, primitives::Add> for Dispatch<Cpu, primitives::Add> {
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

impl Eval<Cpu, primitives::Sub> for Dispatch<Cpu, primitives::Sub> {
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

impl Eval<Cpu, primitives::Mul> for Dispatch<Cpu, primitives::Mul> {
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

impl Eval<Cpu, primitives::Div> for Dispatch<Cpu, primitives::Div> {
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

impl Eval<Cpu, primitives::MatMul> for Dispatch<Cpu, primitives::MatMul> {
    fn eval(&self, _: &Cpu, _: &primitives::MatMul, inputs: &[Tensor], output: &Tensor) {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        let t1 = lhs.get_data::<Data>().unwrap();
        let t2 = rhs.get_data::<Data>().unwrap();
        let t1 = t1.deref();
        let t2 = t2.deref();
        let t = if t1.shape() != t2.shape() {
            t1.broadcast_matmul(t2).unwrap()
        } else {
            t1.matmul(t2).unwrap()
        };
        output.set_data(t);
    }
}

impl Eval<Cpu, primitives::Sin> for Dispatch<Cpu, primitives::Sin> {
    fn eval(&self, _: &Cpu, _: &primitives::Sin, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sin().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Cos> for Dispatch<Cpu, primitives::Cos> {
    fn eval(&self, _: &Cpu, _: &primitives::Cos, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.cos().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Negative> for Dispatch<Cpu, primitives::Negative> {
    fn eval(&self, _: &Cpu, _: &primitives::Negative, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.neg().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::ReduceSum> for Dispatch<Cpu, primitives::ReduceSum> {
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

impl Eval<Cpu, primitives::ReduceMax> for Dispatch<Cpu, primitives::ReduceMax> {
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

impl Eval<Cpu, primitives::ReduceMin> for Dispatch<Cpu, primitives::ReduceMin> {
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

impl Eval<Cpu, primitives::Square> for Dispatch<Cpu, primitives::Square> {
    fn eval(&self, _: &Cpu, _: &primitives::Square, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqr().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Sqrt> for Dispatch<Cpu, primitives::Sqrt> {
    fn eval(&self, _: &Cpu, _: &primitives::Sqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Rsqrt> for Dispatch<Cpu, primitives::Rsqrt> {
    fn eval(&self, _: &Cpu, _: &primitives::Rsqrt, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.sqrt().unwrap().recip().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Transpose> for Dispatch<Cpu, primitives::Transpose> {
    fn eval(&self, _: &Cpu, _: &primitives::Transpose, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.t().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Reshape> for Dispatch<Cpu, primitives::Reshape> {
    fn eval(&self, _: &Cpu, primitive: &primitives::Reshape, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.reshape(primitive.shape()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Broadcast> for Dispatch<Cpu, primitives::Broadcast> {
    fn eval(&self, _: &Cpu, primitive: &primitives::Broadcast, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.broadcast_as(primitive.shape()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Sign> for Dispatch<Cpu, primitives::Sign> {
    fn eval(&self, _: &Cpu, _: &primitives::Sign, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let zero = t.zeros_like().unwrap();
        let t = (t.ge(&zero).unwrap() - t.le(&zero).unwrap()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Abs> for Dispatch<Cpu, primitives::Abs> {
    fn eval(&self, _: &Cpu, _: &primitives::Abs, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.abs().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Exp> for Dispatch<Cpu, primitives::Exp> {
    fn eval(&self, _: &Cpu, _: &primitives::Exp, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.exp().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Log> for Dispatch<Cpu, primitives::Log> {
    fn eval(&self, _: &Cpu, _: &primitives::Log, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = t.log().unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Log2> for Dispatch<Cpu, primitives::Log2> {
    fn eval(&self, _: &Cpu, _: &primitives::Log2, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 2.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Log10> for Dispatch<Cpu, primitives::Log10> {
    fn eval(&self, _: &Cpu, _: &primitives::Log10, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = (t.log().unwrap() / 10.0f64.ln()).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::Equal> for Dispatch<Cpu, primitives::Equal> {
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

impl Eval<Cpu, primitives::NotEqual> for Dispatch<Cpu, primitives::NotEqual> {
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

impl Eval<Cpu, primitives::Greater> for Dispatch<Cpu, primitives::Greater> {
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

impl Eval<Cpu, primitives::GreaterEqual> for Dispatch<Cpu, primitives::GreaterEqual> {
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

impl Eval<Cpu, primitives::Less> for Dispatch<Cpu, primitives::Less> {
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

impl Eval<Cpu, primitives::LessEqual> for Dispatch<Cpu, primitives::LessEqual> {
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

impl Eval<Cpu, primitives::Maximum> for Dispatch<Cpu, primitives::Maximum> {
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
        impl Eval<Cpu, primitives::AsType<$T>> for Dispatch<Cpu, primitives::AsType<$T>> {
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

impl Eval<Cpu, primitives::Softmax> for Dispatch<Cpu, primitives::Softmax> {
    fn eval(&self, _: &Cpu, primitive: &primitives::Softmax, inputs: &[Tensor], output: &Tensor) {
        let x = &inputs[0];
        let t = x.get_data::<Data>().unwrap();
        let t = t.deref();
        let t = softmax(t, primitive.dim).unwrap();
        output.set_data(t)
    }
}

impl Eval<Cpu, primitives::LogSoftmax> for Dispatch<Cpu, primitives::LogSoftmax> {
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
