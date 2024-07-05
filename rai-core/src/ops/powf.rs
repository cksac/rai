use crate::{FloatElemType, Op, Shape, Tensor, Type};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq)]
pub struct Powf<D>
where
    D: Type,
    D::Repr: FloatElemType,
{
    pub exponent: D::Repr,
}

impl<D> Powf<D>
where
    D: Type,
    D::Repr: FloatElemType,
{
    pub fn new(exponent: D::Repr) -> Self {
        Self { exponent }
    }

    pub fn exponent(&self) -> D::Repr {
        self.exponent
    }
}

impl<D> Op for Powf<D>
where
    D: Type,
    D::Repr: FloatElemType,
{
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("PowerFloat")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("PowerFloat({:?})", &self.exponent)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        tangent_x * x.powf(self.exponent - D::one()) * self.exponent
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * x.powf(self.exponent - D::one()) * self.exponent;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn powf<T: FloatElemType>(x: &Tensor, exponent: T) -> Tensor {
    let device = x.device();
    let dtype = x.dtype(); // todo: promote to f64?
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        Powf::<T::DType>::new(exponent),
        inputs,
    )
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn powf<T: FloatElemType>(&self, exponent: T) -> Tensor {
        powf(self, exponent)
    }
}
