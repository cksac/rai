use std::{fmt::Debug, ops::Neg};
use tracing::Level;

use crate::{
    primitives::{
        Abs, Add, Arange, Broadcast, Cos, Div, Exp, Full, Greater, GreaterEqual, Less, LessEqual,
        MatMul, Maximum, Mul, Negative, Normal, ReduceSum, Reshape, Rsqrt, Sign, Sin, Softmax,
        Sqrt, Square, Sub, Transpose,
    },
    shape::Dims,
    utils::dot_graph,
    Backend, DType, Dim, Shape, Tensor,
};

macro_rules! impl_std_ops {
    ($op:ident, $func:ident) => {
        impl std::ops::$op<Tensor> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(&self, &rhs)
            }
        }

        impl std::ops::$op<&Tensor> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Tensor {
                $func(&self, rhs)
            }
        }

        impl std::ops::$op<Tensor> for &Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(self, &rhs)
            }
        }

        impl std::ops::$op<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &Tensor) -> Tensor {
                $func(self, rhs)
            }
        }

        impl<T> std::ops::$op<T> for Tensor
        where
            T: 'static + crate::ElemType,
        {
            type Output = Tensor;

            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like(rhs.into_f64());
                $func(&self, &rhs)
            }
        }

        impl<T> std::ops::$op<T> for &Tensor
        where
            T: 'static + crate::ElemType,
        {
            type Output = Tensor;

            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like(rhs.into_f64());
                $func(self, &rhs)
            }
        }

        impl std::ops::$op<Tensor> for f64 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                let lhs = rhs.full_like(self);
                $func(&lhs, &rhs)
            }
        }

        impl std::ops::$op<&Tensor> for f64 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                let lhs = rhs.full_like(self);
                $func(&lhs, &rhs)
            }
        }

        impl std::ops::$op<Tensor> for f32 {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                let lhs = rhs.full_like(self as f64);
                $func(&lhs, &rhs)
            }
        }

        impl std::ops::$op<&Tensor> for f32 {
            type Output = Tensor;

            fn $func(self, rhs: &Tensor) -> Self::Output {
                let lhs = rhs.full_like(self as f64);
                $func(&lhs, &rhs)
            }
        }
    };
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn full(
    val: f64,
    shape: impl Shape,
    dtype: DType,
    backend: impl Into<Box<dyn Backend>> + Debug,
) -> Tensor {
    let backend = backend.into();
    let inputs = vec![];
    Tensor::new(backend, dtype, shape, Full::new(val), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn normal(
    shape: impl Shape,
    dtype: DType,
    backend: impl Into<Box<dyn Backend>> + Debug,
) -> Tensor {
    let backend = backend.into();
    let inputs = vec![];
    Tensor::new(backend, dtype, shape, Normal, inputs)
}

pub trait ArangeArgs: Debug {
    fn start(&self) -> f64 {
        0.0
    }
    fn stop(&self) -> f64;
    fn step(&self) -> f64 {
        1.0
    }
    fn dtype(&self) -> DType;
}

impl ArangeArgs for f64 {
    fn stop(&self) -> f64 {
        *self
    }

    fn dtype(&self) -> DType {
        DType::F64
    }
}

impl ArangeArgs for (f64, DType) {
    fn stop(&self) -> f64 {
        self.0
    }

    fn dtype(&self) -> DType {
        self.1
    }
}

impl ArangeArgs for (f64, f64) {
    fn start(&self) -> f64 {
        self.0
    }

    fn stop(&self) -> f64 {
        self.1
    }

    fn dtype(&self) -> DType {
        DType::F64
    }
}

impl ArangeArgs for (f64, f64, DType) {
    fn start(&self) -> f64 {
        self.0
    }

    fn stop(&self) -> f64 {
        self.1
    }

    fn dtype(&self) -> DType {
        self.2
    }
}

impl ArangeArgs for (f64, f64, f64) {
    fn start(&self) -> f64 {
        self.0
    }

    fn stop(&self) -> f64 {
        self.1
    }

    fn step(&self) -> f64 {
        self.2
    }

    fn dtype(&self) -> DType {
        DType::F64
    }
}

impl ArangeArgs for (f64, f64, f64, DType) {
    fn start(&self) -> f64 {
        self.0
    }

    fn stop(&self) -> f64 {
        self.1
    }

    fn step(&self) -> f64 {
        self.2
    }

    fn dtype(&self) -> DType {
        self.3
    }
}

impl ArangeArgs for f32 {
    fn stop(&self) -> f64 {
        *self as f64
    }

    fn dtype(&self) -> DType {
        DType::F32
    }
}

impl ArangeArgs for (f32, DType) {
    fn stop(&self) -> f64 {
        self.0 as f64
    }

    fn dtype(&self) -> DType {
        self.1
    }
}

impl ArangeArgs for (f32, f32) {
    fn start(&self) -> f64 {
        self.0 as f64
    }

    fn stop(&self) -> f64 {
        self.1 as f64
    }

    fn dtype(&self) -> DType {
        DType::F32
    }
}

impl ArangeArgs for (f32, f32, DType) {
    fn start(&self) -> f64 {
        self.0 as f64
    }

    fn stop(&self) -> f64 {
        self.1 as f64
    }

    fn dtype(&self) -> DType {
        self.2
    }
}

impl ArangeArgs for (f32, f32, f32) {
    fn start(&self) -> f64 {
        self.0 as f64
    }

    fn stop(&self) -> f64 {
        self.1 as f64
    }

    fn step(&self) -> f64 {
        self.2 as f64
    }

    fn dtype(&self) -> DType {
        DType::F32
    }
}

impl ArangeArgs for (f32, f32, f32, DType) {
    fn start(&self) -> f64 {
        self.0 as f64
    }

    fn stop(&self) -> f64 {
        self.1 as f64
    }

    fn step(&self) -> f64 {
        self.2 as f64
    }

    fn dtype(&self) -> DType {
        self.3
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn arange<T: ArangeArgs>(args: T, backend: impl Into<Box<dyn Backend>> + Debug) -> Tensor {
    let start = args.start();
    let stop = args.stop();
    let step = args.step();
    let dtype = args.dtype();

    let size = std::cmp::max(((stop - start) / step).ceil() as usize, 0);
    let backend = backend.into();
    let inputs = vec![];
    Tensor::new(
        backend,
        dtype,
        [size],
        Arange::new(start, stop, step),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "add({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Add, inputs)
}

impl_std_ops!(Add, add);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "sub({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Sub, inputs)
}

impl_std_ops!(Sub, sub);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "mul({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Mul, inputs)
}

impl_std_ops!(Mul, mul);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "div({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Div, inputs)
}

impl_std_ops!(Div, div);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn negative(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Negative, inputs)
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        negative(&self)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        negative(self)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn square(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Square, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sin(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Sin, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn cos(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Cos, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast_matmul(rhs).unwrap_or_else(|e| {
        panic!(
            "matmul({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });

    let lhs_in = lhs;
    let rhs_in = rhs;

    let mut lhs = lhs.clone();
    let mut rhs = rhs.clone();
    if lhs.ndim() == 1 {
        lhs = lhs.reshape([&[1], lhs.shape()].concat());
    }
    if rhs.ndim() == 1 {
        rhs = rhs.reshape([rhs.shape(), &[1]].concat());
    }
    let inputs = vec![lhs.clone(), rhs.clone()];
    if lhs_in.ndim() == 1 || rhs_in.ndim() == 1 {
        let first = shape.ndim() - if lhs_in.ndim() == 1 { 2 } else { 1 };
        let last = shape.ndim() - if rhs_in.ndim() == 1 { 0 } else { 1 };
        let out_shape = [&shape[..first], &shape[last..]].concat();
        let out = Tensor::new(backend, dtype, shape, MatMul, inputs);
        out.reshape(out_shape)
    } else {
        Tensor::new(backend, dtype, shape, MatMul, inputs)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn transpose(x: &Tensor, dims: impl Into<Vec<usize>> + Debug) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape_transpose();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Transpose::new(dims), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_to(x: &Tensor, shape: impl Shape) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let out_shape = x.shape_broadcast(&shape).unwrap_or_else(|e| {
        panic!(
            "{:?} broadcast_to shape {} with error {:?}\n{}",
            x,
            shape.ndim(),
            e,
            dot_graph(x)
        )
    });
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, out_shape, Broadcast::new(shape), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn reshape(x: &Tensor, shape: impl Shape) -> Tensor {
    if x.shape_eq(&shape) {
        return x.clone();
    }

    if x.shape_size_eq(&shape) {
        let backend = x.backend();
        let dtype = x.dtype();
        let inputs = vec![x.clone()];
        Tensor::new(backend, dtype, shape, Reshape, inputs)
    } else {
        panic!("cannot reshape tensor {:?} to shape {:?}", x, shape.shape());
    }
}

pub trait ReduceSumArgs: Debug {
    fn dims(&self) -> impl Dims;
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> ReduceSumArgs for T
where
    for<'a> &'a T: Dims,
    T: Debug,
{
    fn dims(&self) -> impl Dims {
        self
    }
}

impl<'a, T> ReduceSumArgs for (&'a T, bool)
where
    &'a T: Dims,
    T: Debug,
{
    fn dims(&self) -> impl Dims {
        self.0
    }

    fn keep_dim(&self) -> bool {
        self.1
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn reduce_sum<T: ReduceSumArgs>(x: &Tensor, args: T) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let dims = x.dims(args.dims());
    let shape = x.shape_reduce(&dims, args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        backend,
        dtype,
        shape,
        ReduceSum::new(dims, args.keep_dim()),
        inputs,
    )
}

pub fn sqrt(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Sqrt, inputs)
}

pub fn rsqrt(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Rsqrt, inputs)
}

pub fn sign(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Sign, inputs)
}

pub fn abs(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Abs, inputs)
}

pub fn exp(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Exp, inputs)
}

pub fn greater(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "greater({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });

    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Greater, inputs)
}

pub fn greater_equal(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "greater_equal({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, GreaterEqual, inputs)
}

pub fn less(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "less({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Less, inputs)
}

pub fn less_equal(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "less_equal({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, LessEqual, inputs)
}

pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap_or_else(|e| {
        panic!(
            "maximum({:?}, {:?}) with error {:?}\n{}",
            lhs,
            rhs,
            e,
            dot_graph([lhs, rhs])
        )
    });
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Maximum, inputs)
}

pub fn softmax<T: Dim>(x: &Tensor, dim: T) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dim = shape.dim(dim);
    Tensor::new(backend, dtype, shape, Softmax::new(dim), inputs)
}

pub fn relu(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like())
}
