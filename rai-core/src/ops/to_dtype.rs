use crate::{AsDType, Op, Shape, Tensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ToDType<D: Type> {
    pub dtype: D,
}

impl<D: Type> ToDType<D> {
    pub fn new(dtype: D) -> Self {
        Self { dtype }
    }
}

impl<D: Type> Op for ToDType<D> {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("ToDType({:?})", &self.dtype)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.to_dtype(self.dtype)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.to_dtype(x);
        vec![cotangent_x]
    }
}

#[track_caller]
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
