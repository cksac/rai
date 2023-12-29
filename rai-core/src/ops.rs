use std::{fmt::Debug, ops::Neg};
use tracing::Level;

use crate::{
    primitives::{
        Add, Broadcast, Cos, Div, Full, MatMul, Mul, Negative, Normal, ReduceSum, Reshape, Sin,
        Sqrt, Square, Sub, Transpose,
    },
    Backend, DType, Shape, Tensor,
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
                let rhs = Tensor::full(rhs.into_f64(), vec![], self.dtype(), self.backend());
                $func(&self, &rhs)
            }
        }

        impl<T> std::ops::$op<T> for &Tensor
        where
            T: 'static + crate::ElemType,
        {
            type Output = Tensor;

            fn $func(self, rhs: T) -> Self::Output {
                let rhs = Tensor::full(rhs.into_f64(), vec![], self.dtype(), self.backend());
                $func(self, &rhs)
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

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap();
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Add, inputs)
}

impl_std_ops!(Add, add);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap();
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Sub, inputs)
}

impl_std_ops!(Sub, sub);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap();
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Mul, inputs)
}

impl_std_ops!(Mul, mul);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs.shape_broadcast(rhs).unwrap();
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
    let shape = lhs.shape_broadcast_matmul(rhs).unwrap();

    let lhs_in = lhs;
    let rhs_in = rhs;

    let mut lhs = lhs.clone();
    let mut rhs = rhs.clone();
    if lhs.ndim() == 1 {
        lhs = lhs.reshape([&[1], lhs.dims()].concat());
    }
    if rhs.ndim() == 1 {
        rhs = rhs.reshape([rhs.dims(), &[1]].concat());
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
pub fn transpose(x: &Tensor, axes: impl Into<Vec<usize>> + Debug) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape_transpose();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Transpose::new(axes), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_to(x: &Tensor, shape: impl Shape) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let out_shape = x.shape_broadcast(&shape).expect("broadcast shape");
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
        panic!("cannot reshape tensor");
    }
}

pub trait Axes {
    fn axes(&self) -> &[usize];
    fn to_vec(&self) -> Vec<usize>;
}

impl Axes for Vec<usize> {
    fn axes(&self) -> &[usize] {
        self.as_slice()
    }

    fn to_vec(&self) -> Vec<usize> {
        self.clone()
    }
}

impl Axes for &Vec<usize> {
    fn axes(&self) -> &[usize] {
        self.as_slice()
    }

    fn to_vec(&self) -> Vec<usize> {
        (*self).clone()
    }
}

pub trait ReduceSumArgs: Debug {
    fn axes(&self) -> &[usize];
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> ReduceSumArgs for T
where
    T: Axes + Debug,
{
    fn axes(&self) -> &[usize] {
        Axes::axes(self)
    }
}

impl<T> ReduceSumArgs for (T, bool)
where
    T: Axes + Debug,
{
    fn axes(&self) -> &[usize] {
        self.0.axes()
    }

    fn keep_dim(&self) -> bool {
        self.1
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn reduce_sum<T: ReduceSumArgs>(x: &Tensor, args: T) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();

    let axes = args.axes();
    // TODO: handle keep_dim
    let shape: Vec<usize> = x
        .shape()
        .dims()
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, v)| *v)
        .collect();

    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, ReduceSum::new(axes), inputs)
}

pub fn sqrt(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Sqrt, inputs)
}
