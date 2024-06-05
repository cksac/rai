use crate::{AsDevice, ElemType, Op, Shape, Tensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Normal<D: Type> {
    pub mean: D::Repr,
    pub std: D::Repr,
}

impl<D: Type> Normal<D> {
    pub fn new(mean: D::Repr, std: D::Repr) -> Self {
        Self { mean, std }
    }
}

impl<D: Type> Op for Normal<D> {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Normal({:?}, {:?})", self.mean, self.std)
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
#[track_caller]
pub fn randn<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        shape,
        Normal::<T>::new(T::zero(), T::one()),
        inputs,
    )
}

#[track_caller]
pub fn randn_with<T: ElemType>(
    mean: T,
    std: T,
    shape: impl Shape,
    device: impl AsDevice,
) -> Tensor {
    let dtype = T::DType::boxed_dtype();
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        shape,
        Normal::<T::DType>::new(mean, std),
        inputs,
    )
}

#[track_caller]
pub fn randn_like(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let op = dtype.randn_op();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, op, inputs)
}
