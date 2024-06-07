use crate::{AsDevice, ElemType, Op, RaiResult, Shape, Tensor, TryAsTensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Random<D: Type> {
    pub from: D::Repr,
    pub to: D::Repr,
}

impl<D: Type> Random<D> {
    pub fn new(from: D::Repr, to: D::Repr) -> Self {
        Self { from, to }
    }
}

impl<D: Type> Op for Random<D> {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Random({:?}, {:?})", self.from, self.to)
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
#[track_caller]
pub fn rand<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> RaiResult<Tensor> {
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        shape,
        Random::<T>::new(T::zero(), T::one()),
        inputs,
    )
}

#[track_caller]
pub fn rand_with<T: ElemType>(
    from: T,
    to: T,
    shape: impl Shape,
    device: impl AsDevice,
) -> RaiResult<Tensor> {
    let dtype = T::DType::boxed_dtype();
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        shape,
        Random::<T::DType>::new(from, to),
        inputs,
    )
}

#[track_caller]
pub fn rand_like(x: impl TryAsTensor) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let op = dtype.rand_op();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, op, inputs).into()
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn rand<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> RaiResult<Tensor> {
        rand(shape, dtype, device)
    }

    #[inline]
    #[track_caller]
    pub fn rand_with<T: ElemType>(
        from: T,
        to: T,
        shape: impl Shape,
        device: impl AsDevice,
    ) -> RaiResult<Tensor> {
        rand_with(from, to, shape, device)
    }
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn rand_like(&self) -> RaiResult<Tensor> {
        rand_like(self)
    }
}
