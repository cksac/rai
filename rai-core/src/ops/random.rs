use crate::{AsDevice, ElemType, Op, Shape, Tensor, Type};
use std::{any::Any, borrow::Cow};

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
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(format!("Random<{}>", D::NAME))
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Random({:?}, {:?})", self.from, self.to)
    }

    #[inline]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        output.ones_like()
    }

    #[inline]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        vec![]
    }
}

/// Creates a `Tensor` filled with random values from a uniform distribution on the interval [0, 1).
///
/// Arguments
///
/// * `shape` - The shape of the `Tensor`.
/// * `dtype` - The data type of the `Tensor`.
/// * `device` - The device to place the `Tensor` on.
///
/// Returns
///
/// A `Tensor` filled with random values from a uniform distribution.
#[track_caller]
pub fn rand<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
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
pub fn rand_with<T: ElemType>(from: T, to: T, shape: impl Shape, device: impl AsDevice) -> Tensor {
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
pub fn rand_like(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape();
    let op = dtype.rand_op();
    let inputs = vec![];
    Tensor::new(device, dtype, shape, op, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn rand<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
        rand(shape, dtype, device)
    }

    #[inline]
    #[track_caller]
    pub fn rand_with<T: ElemType>(
        from: T,
        to: T,
        shape: impl Shape,
        device: impl AsDevice,
    ) -> Tensor {
        rand_with(from, to, shape, device)
    }

    #[inline]
    #[track_caller]
    pub fn rand_like(&self) -> Tensor {
        rand_like(self)
    }
}
