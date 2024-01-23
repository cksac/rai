use std::{
    f32::consts::PI,
    fmt::Debug,
    ops::{Neg, Range, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

use tracing::Level;

use crate::{
    primitives::{
        Abs, Add, Arange, ArgMax, ArgMin, AsType, Broadcast, Concatenate, Cos, Div, Equal, Erf,
        Exp, FromArray, Full, Gather, Greater, GreaterEqual, IndexSelect, Less, LessEqual, Log,
        Log10, Log2, LogSoftmax, MatMul, Maximum, Mul, Narrow, Negative, Normal, NotEqual,
        PowerFloat, ReduceMax, ReduceMin, ReduceSum, Reshape, Rsqrt, Sign, Sin, Softmax, Sqrt,
        Square, Sub, Tanh, ToContiguous, Transpose, Where,
    },
    shape::Dims,
    Backend, DType, Dim, DynDType, ElemType, Shape, Tensor, F32, F64, U32, U8,
};

macro_rules! impl_std_ops_for_scalar {
    ($T:ty, $op:ident, $func:ident) => {
        impl std::ops::$op<Tensor> for $T {
            type Output = Tensor;

            fn $func(self, rhs: Tensor) -> Self::Output {
                let lhs = rhs.full_like::<$T>(self);
                $func(&lhs, &rhs)
            }
        }

        impl<'a> std::ops::$op<&'a Tensor> for $T {
            type Output = Tensor;

            fn $func(self, rhs: &'a Tensor) -> Self::Output {
                let lhs = rhs.full_like::<$T>(self);
                $func(&lhs, &rhs)
            }
        }
    };
}

macro_rules! impl_std_ops {
    ($op:ident, $func:ident) => {
        impl std::ops::$op<Tensor> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(&self, &rhs)
            }
        }

        impl<'a> std::ops::$op<&'a Tensor> for Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &'a Tensor) -> Tensor {
                $func(&self, rhs)
            }
        }

        impl<'a> std::ops::$op<Tensor> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(self, &rhs)
            }
        }

        impl<'a, 'b> std::ops::$op<&'b Tensor> for &'a Tensor {
            type Output = Tensor;
            fn $func(self, rhs: &'b Tensor) -> Tensor {
                $func(self, rhs)
            }
        }

        impl<T> std::ops::$op<T> for Tensor
        where
            T: crate::ElemType,
        {
            type Output = Tensor;

            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like::<T>(rhs);
                $func(&self, &rhs)
            }
        }

        impl<'a, T> std::ops::$op<T> for &'a Tensor
        where
            T: crate::ElemType,
        {
            type Output = Tensor;

            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like::<T>(rhs);
                $func(self, &rhs)
            }
        }

        impl_std_ops_for_scalar!(f32, $op, $func);
        impl_std_ops_for_scalar!(f64, $op, $func);
        impl_std_ops_for_scalar!(u8, $op, $func);
    };
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn full<T: ElemType>(
    val: T,
    shape: impl Shape,
    backend: impl Into<Box<dyn Backend>> + Debug,
) -> Tensor {
    let backend = backend.into();
    let inputs = vec![];
    Tensor::new(
        backend,
        T::dyn_dtype(),
        shape,
        Full::<T::DType>::new(val),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn full_like<T: ElemType>(x: &Tensor, val: T) -> Tensor {
    if x.dtype() == T::dyn_dtype().as_ref() {
        full::<T>(val, x.shape(), x.backend())
    } else {
        // TODO: check is type can be convert/promoted to x dtype?
        full::<T>(val, x.shape(), x.backend()).as_type_of(x)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn zeros_like(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape();
    let primitive = dtype.full_zero();
    let inputs = vec![];
    Tensor::new(backend, dtype, shape, primitive, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn ones_like(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape();
    let primitive = dtype.full_one();
    let inputs = vec![];
    Tensor::new(backend, dtype, shape, primitive, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn normal(
    shape: impl Shape,
    dtype: impl DType,
    backend: impl Into<Box<dyn Backend>> + Debug,
) -> Tensor {
    let backend = backend.into();
    let inputs = vec![];
    Tensor::new(backend, dtype, shape, Normal, inputs)
}

pub trait ArangeArgs<D: DType>: Debug {
    fn start(&self) -> D::Repr {
        D::zero()
    }

    fn stop(&self) -> D::Repr;

    fn step(&self) -> D::Repr {
        D::one()
    }

    fn dtype(&self) -> D;

    fn size(&self) -> usize;
}

macro_rules! impl_arange_args {
    ($R:ty, $T:tt) => {
        impl ArangeArgs<$T> for $R {
            fn stop(&self) -> $R {
                *self
            }

            fn dtype(&self) -> $T {
                $T
            }

            fn size(&self) -> usize {
                let start = self.start();
                let stop = self.stop();
                let step = self.step();
                let size = std::cmp::max(((stop - start) / step).ceil() as usize, 0);
                size
            }
        }

        impl ArangeArgs<$T> for ($R, $R) {
            fn start(&self) -> $R {
                self.0
            }

            fn stop(&self) -> $R {
                self.1
            }

            fn dtype(&self) -> $T {
                $T
            }

            fn size(&self) -> usize {
                let start = self.start();
                let stop = self.stop();
                let step = self.step();
                let size = std::cmp::max(((stop - start) / step).ceil() as usize, 0);
                size
            }
        }

        impl ArangeArgs<$T> for ($R, $R, $R) {
            fn start(&self) -> $R {
                self.0
            }

            fn stop(&self) -> $R {
                self.1
            }

            fn step(&self) -> $R {
                self.2
            }

            fn dtype(&self) -> $T {
                $T
            }

            fn size(&self) -> usize {
                let start = self.start();
                let stop = self.stop();
                let step = self.step();
                let size = std::cmp::max(((stop - start) / step).ceil() as usize, 0);
                size
            }
        }
    };

    ($AS:ty, $R:ty, $T:tt) => {
        impl ArangeArgs<$T> for $R {
            fn stop(&self) -> $R {
                *self
            }

            fn dtype(&self) -> $T {
                $T
            }

            fn size(&self) -> usize {
                let start = self.start();
                let stop = self.stop();
                let step = self.step();
                let size = std::cmp::max(((stop - start) as $AS / step as $AS).ceil() as usize, 0);
                size
            }
        }

        impl ArangeArgs<$T> for ($R, $R) {
            fn start(&self) -> $R {
                self.0
            }

            fn stop(&self) -> $R {
                self.1
            }

            fn dtype(&self) -> $T {
                $T
            }

            fn size(&self) -> usize {
                let start = self.start();
                let stop = self.stop();
                let step = self.step();
                let size = std::cmp::max(((stop - start) as $AS / step as $AS).ceil() as usize, 0);
                size
            }
        }

        impl ArangeArgs<$T> for ($R, $R, $R) {
            fn start(&self) -> $R {
                self.0
            }

            fn stop(&self) -> $R {
                self.1
            }

            fn step(&self) -> $R {
                self.2
            }

            fn dtype(&self) -> $T {
                $T
            }

            fn size(&self) -> usize {
                let start = self.start();
                let stop = self.stop();
                let step = self.step();
                let size = std::cmp::max(((stop - start) as $AS / step as $AS).ceil() as usize, 0);
                size
            }
        }
    };
}

impl_arange_args!(f32, F32);
impl_arange_args!(f64, F64);
impl_arange_args!(f32, u8, U8);
impl_arange_args!(f32, u32, U32);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn arange<D: DType, T: ArangeArgs<D>>(
    args: T,
    backend: impl Into<Box<dyn Backend>> + Debug,
) -> Tensor {
    let start = args.start();
    let stop = args.stop();
    let step = args.step();
    let dtype = args.dtype();
    let size = args.size();
    let backend = backend.into();
    let inputs = vec![];
    Tensor::new(
        backend,
        dtype,
        [size],
        Arange::<D>::new(start, stop, step),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn from_array<T: ElemType>(
    data: impl Into<Vec<T>> + Debug,
    shape: impl Shape,
    backend: impl Into<Box<dyn Backend>> + Debug,
) -> Tensor {
    let data = data.into();
    assert!(data.len() == shape.size());
    let backend = backend.into();
    let inputs = vec![];
    Tensor::new(
        backend,
        T::dyn_dtype(),
        shape,
        FromArray::<T::DType>::new(data),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("add({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Add, inputs)
}

impl_std_ops!(Add, add);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("sub({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Sub, inputs)
}

impl_std_ops!(Sub, sub);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("mul({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Mul, inputs)
}

impl_std_ops!(Mul, mul);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("div({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Div, inputs)
}

impl_std_ops!(Div, div);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn neg(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Negative, inputs)
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        neg(&self)
    }
}

impl<'a> Neg for &'a Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        neg(self)
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
pub fn powf(x: &Tensor, exponent: f64) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype(); // todo: promote to f64?
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, PowerFloat::new(exponent), inputs)
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
pub fn tanh(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Tanh, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs
        .shape_broadcast_matmul(rhs)
        .unwrap_or_else(|e| panic!("matmul({:?}, {:?}) with error {:?}", lhs, rhs, e));

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
pub fn transpose(x: &Tensor, dim0: impl Dim, dim1: impl Dim) -> Tensor {
    let dim0 = x.dim(dim0);
    let dim1 = x.dim(dim1);
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape_transpose(dim0, dim1);
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Transpose::new(dim0, dim1), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_to(x: &Tensor, shape: impl Shape) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let out_shape = x.shape_broadcast(&shape).unwrap_or_else(|e| {
        panic!(
            "{:?} broadcast_to shape {} with error {:?}",
            x,
            shape.ndim(),
            e
        )
    });
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, out_shape, Broadcast::new(shape), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_left(x: &Tensor, shape: impl Shape) -> Tensor {
    let out_shape = x.shape_expand_left(&shape);
    x.broadcast_to(out_shape)
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
        Tensor::new(
            backend,
            dtype,
            shape.shape().to_owned(),
            Reshape::new(shape),
            inputs,
        )
    } else {
        panic!("reshape({:?}, {:?}) with error", x, shape.shape());
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sqrt(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Sqrt, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn rsqrt(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Rsqrt, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sign(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Sign, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn abs(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Abs, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn exp(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Exp, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Log, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log2(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Log2, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log10(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Log10, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn eq(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = U8;
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("eq({:?}, {:?}) with error {:?}", lhs, rhs, e));

    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Equal, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn ne(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = U8;
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("ne({:?}, {:?}) with error {:?}", lhs, rhs, e));

    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, NotEqual, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn gt(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = U8;
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("gt({:?}, {:?}) with error {:?}", lhs, rhs, e));

    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Greater, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn ge(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = U8;
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("ge({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, GreaterEqual, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn lt(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = U8;
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("lt({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Less, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn le(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = U8;
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("le({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, LessEqual, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let backend = lhs.backend();
    let dtype = lhs.dtype();
    let shape = lhs
        .shape_broadcast(rhs)
        .unwrap_or_else(|e| panic!("maximum({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = vec![lhs.clone(), rhs.clone()];
    Tensor::new(backend, dtype, shape, Maximum, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn as_type(x: &Tensor, dtype: impl DType) -> Tensor {
    if x.dtype() == &dtype {
        return x.clone();
    }
    let backend = x.backend();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, AsType::new(dtype), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn as_type_of(x: &Tensor, rhs: &Tensor) -> Tensor {
    if x.dtype() == rhs.dtype() {
        return x.clone();
    }
    let backend = x.backend();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dtype: &dyn DynDType = rhs.dtype();
    let primitive = dtype.as_self_dtype();
    Tensor::new(backend, dtype, shape, primitive, inputs)
}

pub trait ReduceArgs: Debug {
    fn dims(&self) -> &impl Dims;
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> ReduceArgs for T
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        self
    }
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> VarArgs for T where T: Dims {}

impl<T> ReduceArgs for (T, bool)
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

impl<T> VarArgs for (T, bool) where T: Dims {}

impl<T> ReduceArgs for (T, usize)
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> VarArgs for (T, usize)
where
    T: Dims,
{
    fn ddof(&self) -> usize {
        self.1
    }
}

impl<T> ReduceArgs for (T, bool, usize)
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

impl<T> VarArgs for (T, bool, usize)
where
    T: Dims,
{
    fn ddof(&self) -> usize {
        self.2
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sum<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
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

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn max<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let dims = x.dims(args.dims());
    let shape = x.shape_reduce(&dims, args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        backend,
        dtype,
        shape,
        ReduceMax::new(dims, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn min<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let dims = x.dims(args.dims());
    let shape = x.shape_reduce(&dims, args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        backend,
        dtype,
        shape,
        ReduceMin::new(dims, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn mean<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of_dims(args.dims()) as f64;
    x.sum(args) / elem_count
}

pub trait VarArgs: ReduceArgs {
    fn ddof(&self) -> usize {
        0
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn var<T: VarArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of_dims(args.dims());
    let m = x.mean((args.dims(), args.keep_dim()));
    let s = (x - m).square().sum((args.dims(), args.keep_dim()));
    s / (elem_count - args.ddof()) as f32
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn softmax<D: Dim>(x: &Tensor, d: D) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dim = shape.dim(d);
    Tensor::new(backend, dtype, shape, Softmax::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log_softmax<D: Dim>(x: &Tensor, d: D) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dim = shape.dim(d);
    Tensor::new(backend, dtype, shape, LogSoftmax::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn erf(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Erf, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn relu(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like())
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn gelu(x: &Tensor) -> Tensor {
    x * 0.5f32 * (1.0f32 + (x / 2.0f32.sqrt()).erf())
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn new_gelu(x: &Tensor) -> Tensor {
    0.5f32 * x * (1.0f32 + ((2.0f32 / PI).sqrt() * (x + 0.044715f32 * x.powf(3.0))).tanh())
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn gather(x: &Tensor, dim: impl Dim, index: &Tensor) -> Tensor {
    assert!(x.shape_eq(index));
    let dim = x.dim(dim);
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(backend, dtype, shape, Gather::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn index_select(x: &Tensor, dim: impl Dim, index: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    let backend = x.backend();
    let dtype = x.dtype();
    let mut shape = x.shape().to_vec();
    shape[dim] = index.size();
    let inputs = vec![x.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(backend, dtype, shape, IndexSelect::new(dim), inputs)
}

pub trait FlattenArgs: Debug {
    fn start_dim(&self) -> impl Dim {
        0
    }
    fn end_dim(&self) -> impl Dim {
        -1
    }
}

impl<D: Dim> FlattenArgs for (D, D) {
    fn start_dim(&self) -> impl Dim {
        &self.0
    }

    fn end_dim(&self) -> impl Dim {
        &self.1
    }
}

impl FlattenArgs for RangeFull {
    fn end_dim(&self) -> impl Dim {
        -1
    }
}

impl FlattenArgs for RangeTo<usize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeTo<isize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeTo<i32> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeToInclusive<usize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeToInclusive<isize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeToInclusive<i32> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for Range<usize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }

    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for Range<isize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }

    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for Range<i32> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }

    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeInclusive<usize> {
    fn start_dim(&self) -> impl Dim {
        self.start()
    }

    fn end_dim(&self) -> impl Dim {
        self.end()
    }
}

impl FlattenArgs for RangeInclusive<isize> {
    fn start_dim(&self) -> impl Dim {
        self.start()
    }

    fn end_dim(&self) -> impl Dim {
        self.end()
    }
}

impl FlattenArgs for RangeInclusive<i32> {
    fn start_dim(&self) -> impl Dim {
        self.start()
    }

    fn end_dim(&self) -> impl Dim {
        self.end()
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn flatten<T: FlattenArgs>(x: &Tensor, args: T) -> Tensor {
    let start_dim = x.dim(args.start_dim());
    let end_dim = x.dim(args.end_dim());
    if x.ndim() == 0 {
        x.reshape([1])
    } else if start_dim < end_dim {
        let mut dst_dim = x.shape_of(..start_dim);
        dst_dim.push(x.size_of_dims(start_dim..=end_dim));
        if end_dim + 1 < x.ndim() {
            dst_dim.extend(x.shape_of(end_dim + 1..));
        }
        x.reshape(dst_dim)
    } else {
        x.clone()
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn squeeze(x: &Tensor, dims: impl Dims) -> Tensor {
    let dims = x.dims(dims).to_vec();
    let mut shape = x.shape().to_vec();
    for d in dims.into_iter() {
        if shape.shape_at(d) == 1 {
            shape.remove(d);
        }
    }
    x.reshape(shape)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn unsqueeze(x: &Tensor, d: impl Dim) -> Tensor {
    let dim = x.dim(d);
    let mut shape = x.shape().to_vec();
    shape.insert(dim, 1);
    x.reshape(shape)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn cat<T: AsRef<Tensor> + Debug>(tensors: &[T], dim: impl Dim) -> Tensor {
    let inputs: Vec<Tensor> = tensors.iter().map(AsRef::as_ref).cloned().collect();
    let t1 = &inputs[0].clone();
    let dim = t1.dim(dim);
    let backend = t1.backend();
    let dtype = t1.dtype();
    let mut shape = t1.shape().to_vec();
    shape[dim] = 0;
    for t in inputs.iter() {
        // todo: check shape
        shape[dim] += t.shape_at(dim);
    }
    Tensor::new(backend, dtype, shape, Concatenate::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn narrow(x: &Tensor, dim: impl Dim, start: usize, len: usize) -> Tensor {
    let dim = x.dim(dim);
    let backend = x.backend();
    let dtype = x.dtype();
    let mut shape = x.shape().to_vec();
    shape[dim] = len;
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, Narrow::new(dim, start, len), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn chunk(x: &Tensor, chunks: usize, dim: impl Dim) -> Vec<Tensor> {
    let dim = x.dim(dim);
    let size = x.shape_at(dim);
    if size < chunks {
        (0..size).map(|i| x.narrow(dim, i, 1)).collect::<Vec<_>>()
    } else {
        let chunk_size = size / chunks;
        let cnt_additional = size % chunks;
        let mut tensors = vec![];
        let mut sum_chunk_size = 0;
        for i in 0..chunks {
            let chunk_size = if i < cnt_additional {
                chunk_size + 1
            } else {
                chunk_size
            };
            let tensor = x.narrow(dim, sum_chunk_size, chunk_size);
            tensors.push(tensor);
            sum_chunk_size += chunk_size
        }
        tensors
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn where_cond(x: &Tensor, input: &Tensor, other: &Tensor) -> Tensor {
    assert_eq!(input.dtype(), other.dtype());
    let backend = x.backend();
    let dtype = input.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone(), input.clone(), other.clone()];
    Tensor::new(backend, dtype, shape, Where, inputs)
}

pub trait ArgReduceArgs: Debug {
    fn dim(&self) -> &impl Dim;
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T: Dim> ArgReduceArgs for T {
    fn dim(&self) -> &impl Dim {
        self
    }
}

impl<T: Dim> ArgReduceArgs for (T, bool) {
    fn dim(&self) -> &impl Dim {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn argmax<T: ArgReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let backend = x.backend();
    let dtype = U32;
    let dim = x.dim(args.dim());
    let shape = x.shape_reduce([dim], args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        backend,
        dtype,
        shape,
        ArgMax::new(dim, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn argmin<T: ArgReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let backend = x.backend();
    let dtype = U32;
    let dim = x.dim(args.dim());
    let shape = x.shape_reduce([dim], args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        backend,
        dtype,
        shape,
        ArgMin::new(dim, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn to_contiguous(x: &Tensor) -> Tensor {
    let backend = x.backend();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(backend, dtype, shape, ToContiguous, inputs)
}
