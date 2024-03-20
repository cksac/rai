use crate::{
    primitives::{
        Abs, Add, Arange, ArgMax, ArgMin, Broadcast, Concatenate, Conv1d, Conv2d, ConvTranspose1d,
        ConvTranspose2d, Cos, Div, Equal, Erf, Exp, FlashAttention, FromArray, Full, Gather,
        Greater, GreaterEqual, IndexAdd, IndexSelect, Less, LessEqual, Log, Log10, Log2,
        LogSoftmax, MatMul, MaxPool2d, Maximum, Minimum, Mul, Narrow, Negative, Normal, NotEqual,
        Permute, PowerFloat, Random, ReduceMax, ReduceMin, ReduceSum, Reshape, Rsqrt, ScatterAdd,
        Sign, Sin, Softmax, Sqrt, Square, Sub, Tanh, ToContiguous, Transpose, Where,
    },
    shape::Dims,
    AsDType, AsDevice, Dim, ElemType, Shape, Tensor, Type, F16, F32, F64, U32, U8,
};
use half::{bf16, f16};
use safetensors::tensor::TensorView;
use std::{
    f32::consts::PI,
    fmt::Debug,
    ops::{Neg, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
    slice::from_raw_parts,
};
use tracing::Level;

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

macro_rules! broadcast_binary_op {
    ($(#[$meta:meta])* $primitive:ident, $func:ident) => {
        $(#[$meta])*
        #[tracing::instrument(ret(level = Level::TRACE))]
        pub fn $func(lhs: &Tensor, rhs: &Tensor) -> Tensor {
            let device = lhs.device();
            let dtype = lhs.dtype();
            let (shape, lhs_b, rhs_b) = lhs.shape_broadcast_to(rhs).unwrap_or_else(|e| {
                panic!(
                    "{}({:?}, {:?}) with error {:?}",
                    stringify!($func),
                    lhs,
                    rhs,
                    e
                )
            });
            let inputs = match (lhs_b, rhs_b) {
                (false, false) => vec![lhs.clone(), rhs.clone()],
                (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&shape)],
                (true, false) => vec![lhs.broadcast_to_unchecked(&shape), rhs.clone()],
                (true, true) => vec![lhs.broadcast_to_unchecked(&shape), rhs.broadcast_to_unchecked(&shape)],
            };
            Tensor::new(device, dtype, shape, $primitive, inputs)
        }
    };
    ($(#[$meta:meta])* $primitive:ident, $func:ident, $out_ty:ident) => {
        $(#[$meta])*
        #[tracing::instrument(ret(level = Level::TRACE))]
        pub fn $func(lhs: &Tensor, rhs: &Tensor) -> Tensor {
            let device = lhs.device();
            let dtype = $out_ty;
            let (shape, lhs_b, rhs_b) = lhs.shape_broadcast_to(rhs).unwrap_or_else(|e| {
                panic!(
                    "{}({:?}, {:?}) with error {:?}",
                    stringify!($func),
                    lhs,
                    rhs,
                    e
                )
            });
            let inputs = match (lhs_b, rhs_b) {
                (false, false) => vec![lhs.clone(), rhs.clone()],
                (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&shape)],
                (true, false) => vec![lhs.broadcast_to_unchecked(&shape), rhs.clone()],
                (true, true) => vec![lhs.broadcast_to_unchecked(&shape), rhs.broadcast_to_unchecked(&shape)],
            };
            Tensor::new(device, dtype, shape, $primitive, inputs)
        }
    };
}

/// Creates a `Tensor` filled with a specified value.
///
/// # Arguments
///
/// * `val` - The value to fill the `Tensor` with.
/// * `shape` - The shape of the `Tensor`.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` filled with the specified value.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn full<T: ElemType>(val: T, shape: impl Shape, device: impl AsDevice) -> Tensor {
    let inputs = vec![];
    Tensor::new(
        device,
        T::DType::boxed_dtype(),
        shape,
        Full::<T::DType>::new(val),
        inputs,
    )
}

/// Creates a `Tensor` filled with ones.
///
/// # Arguments
///
/// * `shape` - The shape of the `Tensor`.
/// * `dtype` - The data type of the `Tensor`.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` filled with ones.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn ones(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
    let dtype = dtype.dtype();
    let primitive = dtype.primitive_full_one();
    Tensor::new(device, dtype, shape, primitive, vec![])
}

/// Creates a `Tensor` filled with zeros.
///
/// # Arguments
///
/// * `shape` - The shape of the `Tensor`.
/// * `dtype` - The data type of the `Tensor`.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` filled with zeros.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn zeros(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
    let dtype = dtype.dtype();
    let primitive = dtype.primitive_full_zero();
    Tensor::new(device, dtype, shape, primitive, vec![])
}

/// Creates a `Tensor` filled with a specified value, with the same shape, data type and device as another `Tensor`.
///
/// # Arguments
///
/// * `x` - The reference to the `Tensor` to mimic the shape, data type and device.
/// * `val` - The value to fill the `Tensor` with.
///
/// # Returns
///
/// A `Tensor` filled with the specified value, with the same shape, data type and device as `x`.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn full_like<T: ElemType>(x: &Tensor, val: T) -> Tensor {
    if x.dtype() == T::DType::boxed_dtype().as_ref() {
        full::<T>(val, x.shape(), x.device())
    } else {
        // TODO: check if type can be converted/promoted to x dtype?
        full::<T>(val, x.shape(), x.device()).to_dtype(x)
    }
}

/// Creates a `Tensor` filled with zeros, with the same shape, data type and device as another `Tensor`.
///
/// # Arguments
///
/// * `x` - The reference to the `Tensor` to mimic the shape, data type and device.
///
/// # Returns
///
/// A `Tensor` filled with zeros, with the same shape, data type and device as `x`.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn zeros_like(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let primitive = dtype.primitive_full_zero();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, primitive, inputs)
}

/// Creates a `Tensor` filled with ones, with the same shape, data type and device as another `Tensor`.
///
/// # Arguments
///
/// * `x` - The reference to the `Tensor` to mimic the shape, data type and device.
///
/// # Returns
///
/// A `Tensor` filled with ones, with the same shape, data type and device as `x`.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn ones_like(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let primitive = dtype.primitive_full_one();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, primitive, inputs)
}

/// Creates a `Tensor` filled with random values from a normal distribution with mean 0 and variance 1.
///
/// # Arguments
///
/// * `shape` - The shape of the `Tensor`.
/// * `dtype` - The data type of the `Tensor`.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` filled with random values from a normal distribution.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn randn<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
    let dtype = T::boxed_dtype();
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        shape,
        Normal::<T>::new(T::zero(), T::one()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn randn_with<T: Type>(
    mean: T::Repr,
    std: T::Repr,
    shape: impl Shape,
    dtype: T,
    device: impl AsDevice,
) -> Tensor {
    let dtype = T::boxed_dtype();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, Normal::<T>::new(mean, std), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn randn_like(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let primitive = dtype.primitive_randn();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, primitive, inputs)
}

/// Creates a `Tensor` filled with random values from a uniform distribution on the interval [0, 1).
///
/// # Arguments
///
/// * `shape` - The shape of the `Tensor`.
/// * `dtype` - The data type of the `Tensor`.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` filled with random values from a uniform distribution.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn rand<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
    let dtype = T::boxed_dtype();
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        shape,
        Random::<T>::new(T::zero(), T::one()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn rand_with<T: Type>(
    from: T::Repr,
    to: T::Repr,
    shape: impl Shape,
    dtype: T,
    device: impl AsDevice,
) -> Tensor {
    let dtype = T::boxed_dtype();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, Random::<T>::new(from, to), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn rand_like(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let primitive = dtype.primitive_rand();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, primitive, inputs)
}

/// Represents the arguments for the `arange` function.
pub trait ArangeArgs<D: Type>: Debug {
    /// Returns the start value for the `arange` function.
    fn start(&self) -> D::Repr {
        D::Repr::zero()
    }

    /// Returns the stop value for the `arange` function.
    fn stop(&self) -> D::Repr;

    /// Returns the step value for the `arange` function.
    fn step(&self) -> D::Repr {
        D::Repr::one()
    }

    /// Returns the size of the resulting `Tensor` for the `arange` function.
    fn size(&self) -> usize {
        D::Repr::elem_count(self.start(), self.stop(), self.step())
    }
}

macro_rules! impl_arange_args {
    ($R:ty, $T:tt) => {
        impl ArangeArgs<$T> for $R {
            fn stop(&self) -> $R {
                *self
            }
        }

        impl ArangeArgs<$T> for ($R, $R) {
            fn start(&self) -> $R {
                self.0
            }

            fn stop(&self) -> $R {
                self.1
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
        }
    };
}

impl_arange_args!(f32, F32);
impl_arange_args!(f64, F64);
impl_arange_args!(f16, F16);
impl_arange_args!(u8, U8);
impl_arange_args!(u32, U32);

/// Creates a 1-D `Tensor` with values from a range.
///
/// # Arguments
///
/// * `args` - The arguments for the `arange` function.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A 1-D `Tensor` with values from the specified range.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn arange<D: Type, T: ArangeArgs<D>>(args: T, device: impl AsDevice) -> Tensor {
    let start = args.start();
    let stop = args.stop();
    let step = args.step();
    let dtype = D::boxed_dtype();
    let size = args.size();
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        [size],
        Arange::<D>::new(start, stop, step),
        inputs,
    )
}

/// Creates a `Tensor` from an array of values.
///
/// # Arguments
///
/// * `data` - The array of values.
/// * `shape` - The shape of the `Tensor`.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` created from the array of values.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn from_array<T: ElemType>(
    data: impl Into<Vec<T>> + Debug,
    shape: impl Shape,
    device: impl AsDevice,
) -> Tensor {
    let data = data.into();
    assert!(data.len() == shape.size());
    let inputs = vec![];
    Tensor::new(
        device,
        T::DType::boxed_dtype(),
        shape,
        FromArray::<T::DType>::new(data),
        inputs,
    )
}

// Note: modified from candle candle_core::safetensors::convert_slice
/// Converts a byte slice to a typed slice.
///
/// # Arguments
///
/// * `data` - The byte slice to convert.
///
/// # Returns
///
/// A typed slice converted from the byte slice.
fn convert_slice<T: Clone>(data: &[u8]) -> Vec<T> {
    let size_in_bytes = std::mem::size_of::<T>();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY: This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] = unsafe { from_raw_parts(data.as_ptr() as *const T, elem_count) };
        data.to_vec()
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non-overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        c
    }
}

/// Creates a `Tensor` from a `safetensors::TensorView`.
///
/// # Arguments
///
/// * `view` - The `safetensors::TensorView` to create the `Tensor` from.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` created from the `safetensors::TensorView`.
#[tracing::instrument(ret(level = Level::TRACE))]
pub fn from_safetensor(view: &TensorView, device: impl AsDevice) -> Tensor {
    let shape = view.shape();
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::BOOL => todo!(),
        safetensors::Dtype::U8 => {
            let data = convert_slice::<u8>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::I8 => todo!(),
        safetensors::Dtype::I16 => todo!(),
        safetensors::Dtype::U16 => todo!(),
        safetensors::Dtype::F16 => {
            let data = convert_slice::<f16>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::BF16 => {
            let data = convert_slice::<bf16>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::I32 => todo!(),
        safetensors::Dtype::U32 => {
            let data = convert_slice::<u32>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::F32 => {
            let data = convert_slice::<f32>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::F64 => {
            let data = convert_slice::<f64>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::I64 => {
            let data = convert_slice::<i64>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::U64 => todo!(),
        _ => todo!(),
    }
}

broadcast_binary_op!(
    /// Adds two `Tensor` objects.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the addition.
    Add,
    add
);

broadcast_binary_op!(
    /// Subtracts two `Tensor` objects.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the subtraction.
    Sub,
    sub
);

broadcast_binary_op!(
    /// Multiplies two `Tensor` objects.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the multiplication.
    Mul,
    mul
);

broadcast_binary_op!(
    /// Divides two `Tensor` objects.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The first `Tensor`.
    /// * `rhs` - The second `Tensor`.
    ///
    /// # Returns
    ///
    /// The resulting `Tensor` after the division.
    Div,
    div
);

broadcast_binary_op!(Equal, eq, U8);
broadcast_binary_op!(NotEqual, ne, U8);
broadcast_binary_op!(Greater, gt, U8);
broadcast_binary_op!(GreaterEqual, ge, U8);
broadcast_binary_op!(Less, lt, U8);
broadcast_binary_op!(LessEqual, le, U8);
broadcast_binary_op!(Maximum, maximum);
broadcast_binary_op!(Minimum, minimum);

impl_std_ops!(Add, add);
impl_std_ops!(Sub, sub);
impl_std_ops!(Mul, mul);
impl_std_ops!(Div, div);

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn neg(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Negative, inputs)
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
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Square, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn powf(x: &Tensor, exponent: f64) -> Tensor {
    let device = x.device();
    let dtype = x.dtype(); // todo: promote to f64?
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, PowerFloat::new(exponent), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sin(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Sin, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn cos(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Cos, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn tanh(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Tanh, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let device = lhs.device();
    let dtype = lhs.dtype();
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
    let (mut shape, lhs_shape, rhs_shape, lhs_b, rhs_b) = lhs
        .shape_broadcast_matmul(&rhs)
        .unwrap_or_else(|e| panic!("matmul({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = match (lhs_b, rhs_b) {
        (false, false) => vec![lhs.clone(), rhs.clone()],
        (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&rhs_shape)],
        (true, false) => vec![lhs.broadcast_to_unchecked(&lhs_shape), rhs.clone()],
        (true, true) => vec![
            lhs.broadcast_to_unchecked(&lhs_shape),
            rhs.broadcast_to_unchecked(&rhs_shape),
        ],
    };
    if lhs_in.ndim() == 1 || rhs_in.ndim() == 1 {
        let erase_start = shape.len() - if lhs_in.ndim() == 1 { 2 } else { 1 };
        let erase_end = shape.len() - if rhs_in.ndim() == 1 { 0 } else { 1 };
        let matml_out = Tensor::new(device, dtype, &shape, MatMul, inputs);
        shape.drain(erase_start..erase_end);
        matml_out.reshape(shape)
    } else {
        Tensor::new(device, dtype, shape, MatMul, inputs)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn transpose(x: &Tensor, dim0: impl Dim, dim1: impl Dim) -> Tensor {
    let dim0 = x.dim(dim0);
    let dim1 = x.dim(dim1);
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape_transpose(dim0, dim1);
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Transpose::new(dim0, dim1), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_to(x: &Tensor, shape: impl Shape) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let (out_shape, _, _) = x.shape_broadcast_to(&shape).unwrap_or_else(|e| {
        panic!(
            "{:?} broadcast_to shape {:?} with error {:?}",
            x,
            shape.shape(),
            e
        )
    });
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, out_shape, Broadcast::new(shape), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_to_unchecked(x: &Tensor, shape: impl Shape) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, &shape, Broadcast::new(&shape), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_left(x: &Tensor, shape: impl Shape) -> Tensor {
    let out_shape = x.shape_expand_left(&shape);
    x.broadcast_to_unchecked(out_shape)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn broadcast_right(x: &Tensor, shape: impl Shape) -> Tensor {
    let out_shape = x.shape_expand_right(&shape);
    let mut x = x.clone();
    for _ in x.ndim()..out_shape.ndim() {
        x = x.unsqueeze(-1);
    }
    x.broadcast_to_unchecked(out_shape)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn reshape(x: &Tensor, shape: impl Shape) -> Tensor {
    if x.shape_eq(&shape) {
        return x.clone();
    }

    if x.shape_size_eq(&shape) {
        let device = x.device();
        let dtype = x.dtype();
        let inputs = vec![x.clone()];
        Tensor::new(
            device,
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
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Sqrt, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn rsqrt(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Rsqrt, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn sign(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Sign, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn abs(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Abs, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn exp(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Exp, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Log, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log2(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Log2, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log10(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Log10, inputs)
}

pub trait ClampBound: Debug {
    fn bound(&self, input: &Tensor) -> Tensor;
}

impl<T: ElemType> ClampBound for T {
    fn bound(&self, input: &Tensor) -> Tensor {
        input.full_like(*self).to_dtype(input)
    }
}

impl ClampBound for Tensor {
    fn bound(&self, input: &Tensor) -> Tensor {
        self.to_dtype(input)
    }
}

impl ClampBound for &Tensor {
    fn bound(&self, input: &Tensor) -> Tensor {
        (*self).to_dtype(input)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn clamp(x: &Tensor, min: impl ClampBound, max: impl ClampBound) -> Tensor {
    let min = min.bound(x);
    let max = max.bound(x);
    x.maximum(min).minimum(max)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn to_dtype(x: &Tensor, dtype: impl AsDType) -> Tensor {
    let dtype = dtype.dtype();
    if x.dtype() == dtype {
        return x.clone();
    }
    let device = x.device();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let primitive = dtype.primitive_as_dtype();
    Tensor::new(device, dtype, shape, primitive, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn to_device(x: &Tensor, device: impl AsDevice) -> Tensor {
    let device = device.device();
    if x.device() == device {
        return x.clone();
    }
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let primitive = device.primitive_to_device();
    Tensor::new(device, dtype, shape, primitive, inputs)
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
    let device = x.device();
    let dtype = x.dtype();
    let dims = x.dims(args.dims());
    let shape = x.shape_reduce(&dims, args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ReduceSum::new(dims, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn max<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let dims = x.dims(args.dims());
    let shape = x.shape_reduce(&dims, args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ReduceMax::new(dims, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn min<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let dims = x.dims(args.dims());
    let shape = x.shape_reduce(&dims, args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ReduceMin::new(dims, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn mean<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of(args.dims()) as f64;
    x.sum(args) / elem_count
}

pub trait VarArgs: ReduceArgs {
    fn ddof(&self) -> usize {
        0
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn var<T: VarArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of(args.dims());
    let m = x.mean((args.dims(), args.keep_dim()));
    let s = (x - m).square().sum((args.dims(), args.keep_dim()));
    s / (elem_count - args.ddof()) as f32
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn softmax<D: Dim>(x: &Tensor, d: D) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dim = shape.dim(d);
    Tensor::new(device, dtype, shape, Softmax::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn log_softmax<D: Dim>(x: &Tensor, d: D) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dim = shape.dim(d);
    Tensor::new(device, dtype, shape, LogSoftmax::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn erf(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Erf, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn relu(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like())
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn relu2(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like()).sqrt()
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn relu6(x: &Tensor) -> Tensor {
    x.clamp(0.0f32, 6.0f32)
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
pub fn silu(x: &Tensor) -> Tensor {
    x / (x.neg().exp() + 1.0f32)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn gather(x: &Tensor, dim: impl Dim, index: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    assert_eq!(x.ndim(), index.ndim());
    let mut lhs_shape = x.shape().to_vec();
    lhs_shape.remove(dim);
    let mut idx_shape = index.shape().to_vec();
    idx_shape.remove(dim);
    assert_eq!(lhs_shape, idx_shape);
    let device = x.device();
    let dtype = x.dtype();
    let shape = index.shape();
    let inputs = vec![x.clone(), index.clone()];
    Tensor::new(device, dtype, shape, Gather::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn index_add(x: &Tensor, dim: impl Dim, index: &Tensor, source: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    // due to vjp only return by position
    // x and source will have grads, therefore it comes first
    let inputs = vec![x.clone(), source.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(device, dtype, shape, IndexAdd::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn index_select(x: &Tensor, dim: impl Dim, index: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let mut shape = x.shape().to_vec();
    shape[dim] = index.size();
    let inputs = vec![x.clone(), index.clone()];
    // TODO: asserts
    Tensor::new(device, dtype, shape, IndexSelect::new(dim), inputs)
}

pub trait FlattenArgs: Debug {
    fn start_dim(&self) -> impl Dim {
        0
    }
    fn end_dim(&self) -> impl Dim {
        -1
    }
}

impl FlattenArgs for usize {
    fn start_dim(&self) -> impl Dim {
        self
    }
}

impl FlattenArgs for isize {
    fn start_dim(&self) -> impl Dim {
        self
    }
}

impl FlattenArgs for i32 {
    fn start_dim(&self) -> impl Dim {
        self
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

impl FlattenArgs for RangeFrom<usize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }
}

impl FlattenArgs for RangeFrom<isize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }
}

impl FlattenArgs for RangeFrom<i32> {
    fn start_dim(&self) -> impl Dim {
        self.start
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
    if x.ndim() == 0 {
        return x.reshape([1]);
    }
    let start_dim = x.dim(args.start_dim());
    let end_dim = x.dim(args.end_dim());
    if start_dim < end_dim {
        let mut dst_dim = x.shape_of(..start_dim);
        dst_dim.push(x.size_of(start_dim..=end_dim));
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
    let mut out_shape = Vec::new();
    for (i, s) in x.shape().iter().enumerate() {
        if !dims.contains(&i) || *s != 1 {
            out_shape.push(*s);
        }
    }
    x.reshape(out_shape)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn unsqueeze(x: &Tensor, d: impl Dim) -> Tensor {
    let is_negative = d.is_negative();
    let dim = x.dim(d);
    let mut shape = x.shape().to_vec();
    if is_negative {
        shape.insert(dim + 1, 1);
    } else {
        shape.insert(dim, 1);
    }
    x.reshape(shape)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn permute(x: &Tensor, d: impl Dims) -> Tensor {
    let dims = x.dims(d);
    assert_eq!(dims.len(), x.ndim());
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape_of(&dims);
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Permute::new(dims), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn cat<T: AsRef<Tensor> + Debug>(tensors: &[T], dim: impl Dim) -> Tensor {
    let inputs: Vec<Tensor> = tensors.iter().map(AsRef::as_ref).cloned().collect();
    let t1 = &inputs[0].clone();
    let dim = t1.dim(dim);
    let device = t1.device();
    let dtype = t1.dtype();
    let mut shape = t1.shape().to_vec();
    shape[dim] = 0;
    for t in inputs.iter() {
        // todo: check shape
        shape[dim] += t.shape_at(dim);
    }
    Tensor::new(device, dtype, shape, Concatenate::new(dim), inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn narrow(x: &Tensor, dim: impl Dim, start: usize, len: usize) -> Tensor {
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let mut shape = x.shape().to_vec();
    shape[dim] = len;
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Narrow::new(dim, start, len), inputs)
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
    let device = x.device();
    let dtype = input.dtype();
    let shape = x.shape();
    // no grad for x, therefore, it goes last in input list
    let inputs = vec![input.clone(), other.clone(), x.clone()];
    Tensor::new(device, dtype, shape, Where, inputs)
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
    let device = x.device();
    let dtype = U32;
    let dim = x.dim(args.dim());
    let shape = x.shape_reduce([dim], args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ArgMax::new(dim, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn argmin<T: ArgReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let device = x.device();
    let dtype = U32;
    let dim = x.dim(args.dim());
    let shape = x.shape_reduce([dim], args.keep_dim());
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ArgMin::new(dim, args.keep_dim()),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn to_contiguous(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, ToContiguous, inputs)
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn scatter_add(x: &Tensor, dim: impl Dim, index: &Tensor, source: &Tensor) -> Tensor {
    let dim = x.dim(dim);
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let inputs = vec![x.clone(), source.clone(), index.clone()];
    Tensor::new(device, dtype, shape, ScatterAdd::new(dim), inputs)
}

pub trait FlashAttentionOpts: Debug {
    fn softmax_scale(&self) -> f32;
    fn window_size_left(&self) -> Option<usize> {
        None
    }
    fn window_size_right(&self) -> Option<usize> {
        None
    }
    fn alibi_slopes(&self) -> Option<&Tensor> {
        None
    }
}

impl FlashAttentionOpts for f32 {
    fn softmax_scale(&self) -> f32 {
        *self
    }
}

impl<'a> FlashAttentionOpts for (f32, &'a Tensor) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn alibi_slopes(&self) -> Option<&Tensor> {
        Some(self.1)
    }
}

impl<'a> FlashAttentionOpts for (f32, usize, &'a Tensor) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn window_size_right(&self) -> Option<usize> {
        Some(self.1)
    }

    fn alibi_slopes(&self) -> Option<&Tensor> {
        Some(self.2)
    }
}

impl FlashAttentionOpts for (f32, usize, usize) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn window_size_right(&self) -> Option<usize> {
        Some(self.1)
    }

    fn window_size_left(&self) -> Option<usize> {
        Some(self.2)
    }
}

impl<'a> FlashAttentionOpts for (f32, usize, usize, &'a Tensor) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn window_size_right(&self) -> Option<usize> {
        Some(self.1)
    }

    fn window_size_left(&self) -> Option<usize> {
        Some(self.2)
    }

    fn alibi_slopes(&self) -> Option<&Tensor> {
        Some(self.3)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    opts: impl FlashAttentionOpts,
) -> Tensor {
    let device = q.device();
    let dtype = q.dtype();
    let shape = q.shape();
    let inputs = vec![q.clone(), k.clone(), v.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        FlashAttention::new(
            opts.softmax_scale(),
            opts.window_size_left(),
            opts.window_size_right(),
            opts.alibi_slopes().cloned(),
        ),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
fn conv1d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv1d(kernel, padding, stride, dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        Conv1d::new(padding, stride, dilation),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn conv1d(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let c_in = input.shape_at(1);
    let c_in_k = kernel.shape_at(1);
    assert_eq!(c_in, c_in_k * groups);
    if groups == 1 {
        conv1d_single_group(input, kernel, padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| conv1d_single_group(block, kernel, padding, stride, dilation))
            .collect::<Vec<_>>();
        cat(&outputs, 1)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
fn conv2d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv2d(kernel, &padding, &stride, &dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        Conv2d::new(padding, stride, dilation),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Tensor {
    let c_in = input.shape_at(1);
    let c_in_k = kernel.shape_at(1);
    assert_eq!(c_in, c_in_k * groups);
    if groups == 1 {
        conv2d_single_group(input, kernel, padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| conv2d_single_group(block, kernel, padding, stride, dilation))
            .collect::<Vec<_>>();
        cat(&outputs, 1)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
fn conv_transpose1d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv_transpose1d(kernel, padding, output_padding, stride, dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ConvTranspose1d::new(padding, output_padding, stride, dilation),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn conv_transpose1d(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    if groups == 1 {
        conv_transpose1d_single_group(input, kernel, padding, output_padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| {
                conv_transpose1d_single_group(
                    block,
                    kernel,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                )
            })
            .collect::<Vec<_>>();
        cat(&outputs, 1)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
fn conv_transpose2d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    output_padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv_transpose2d(kernel, &padding, &output_padding, &stride, &dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ConvTranspose2d::new(padding, output_padding, stride, dilation),
        inputs,
    )
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn conv_transpose2d(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    output_padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Tensor {
    if groups == 1 {
        conv_transpose2d_single_group(input, kernel, padding, output_padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| {
                conv_transpose2d_single_group(
                    block,
                    kernel,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                )
            })
            .collect::<Vec<_>>();
        cat(&outputs, 1)
    }
}

#[tracing::instrument(ret(level = Level::TRACE))]
pub fn max_pool2d(
    input: &Tensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_max_pool2d(&kernel_size, &stride, &padding, &dilation);
    let inputs = vec![input.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        MaxPool2d::new(kernel_size, stride, padding, dilation),
        inputs,
    )
}
