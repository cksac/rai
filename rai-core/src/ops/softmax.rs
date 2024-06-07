use crate::{Dim, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Softmax {
    pub dim: usize,
}

impl Softmax {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Op for Softmax {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Softmax({:?})", &self.dim)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        let sv = &(output * tangent_x);
        sv - output * sv.sum((self.dim, true))
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let sv = &(output * cotangent);
        let cotangent_x = sv - output * sv.sum((self.dim, true));
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn softmax<D: Dim>(x: impl TryAsTensor, d: D) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let dim = shape.dim(d);
    Tensor::new(device, dtype, shape, Softmax::new(dim), inputs).into()
}

pub trait SoftmaxOp {
    fn softmax<D: Dim>(self, d: D) -> RaiResult<Tensor>;
}

impl<T> SoftmaxOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn softmax<D: Dim>(self, d: D) -> RaiResult<Tensor> {
        softmax(self, d)
    }
}
