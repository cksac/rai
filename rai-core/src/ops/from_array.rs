use crate::{AsDevice, ElemType, FloatElemType, Op, Shape, Tensor, Type};
use std::{any::Any, fmt::Debug};
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct FromArray<D>
where
    D: Type,
{
    pub data: Vec<D::Repr>,
}

impl<D> FromArray<D>
where
    D: Type,
{
    pub fn new(data: impl Into<Vec<D::Repr>>) -> Self {
        Self { data: data.into() }
    }
}

impl<D> Op for FromArray<D>
where
    D: Type,
{
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        "FromArray(...)".to_string()
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
#[track_caller]
pub fn from_array<T: ElemType>(
    data: impl Into<Vec<T>> + Debug,
    shape: impl Shape,
    device: impl AsDevice,
) -> Tensor {
    let data = data.into();
    assert!(data.len() == shape.elem_count());
    let inputs = vec![];
    Tensor::new(
        device,
        T::DType::boxed_dtype(),
        shape,
        FromArray::<T::DType>::new(data),
        inputs,
    )
}

#[track_caller]
pub fn linspace<T: FloatElemType>(start: T, end: T, steps: usize, device: impl AsDevice) -> Tensor {
    let data = T::linspace(start, end, steps);
    from_array(data, [steps], device)
}
