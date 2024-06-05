use crate::{AsDType, AsDevice, ElemType, Op, Shape, Tensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Full<D>
where
    D: Type,
{
    pub val: D::Repr,
}
impl<D> Full<D>
where
    D: Type,
{
    pub fn new(val: D::Repr) -> Self {
        Full { val }
    }
}

impl<D> Op for Full<D>
where
    D: Type,
{
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn dot_label(&self) -> String {
        format!("Full({:?})", self.val)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    #[inline]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        output.ones_like()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    #[inline]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        vec![]
    }
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
#[track_caller]
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
#[track_caller]
pub fn full_like<T: ElemType>(x: &Tensor, val: T) -> Tensor {
    if x.dtype() == T::DType::boxed_dtype().as_ref() {
        full::<T>(val, x.shape(), x.device())
    } else {
        // TODO: check if type can be converted/promoted to x dtype?
        full::<T>(val, x.shape(), x.device()).to_dtype(x)
    }
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
#[track_caller]
pub fn ones(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
    let dtype = dtype.dtype();
    let device = device.device();
    let op = dtype.full_one_op();
    Tensor::new(device, dtype, shape, op, vec![])
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
#[track_caller]
pub fn ones_like(x: &Tensor) -> Tensor {
    let dtype = x.dtype();
    let device = x.device();
    let shape = x.shape();
    let op = dtype.full_one_op();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, op, inputs)
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
#[track_caller]
pub fn zeros(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
    let dtype = dtype.dtype();
    let device = device.device();
    let op = dtype.full_zero_op();
    Tensor::new(device, dtype, shape, op, vec![])
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
#[track_caller]
pub fn zeros_like(x: &Tensor) -> Tensor {
    let dtype = x.dtype();
    let shape = x.shape();
    let device = x.device();
    let op = dtype.full_zero_op();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, op, inputs)
}
