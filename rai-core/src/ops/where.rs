use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Where;

impl Op for Where {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Where")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let pred = &primals[2];
        let tangent_t = &tangents[0];
        let tangent_f = &tangents[1];
        pred.where_cond(tangent_t, tangent_f)
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let pred = &primals[2];
        let zeros = &cotangent.zeros_like();
        let contangent_t = pred.where_cond(cotangent, zeros);
        let contangent_f = pred.where_cond(zeros, cotangent);
        vec![contangent_t, contangent_f]
    }
}

#[track_caller]
pub fn where_cond(x: &Tensor, input: &Tensor, other: &Tensor) -> Tensor {
    assert_eq!(input.dtype(), other.dtype());
    let device = x.device();
    let dtype = input.dtype();
    let shape = x.shape();
    // no grad for x, therefore, it goes last in input list
    let inputs = vec![input.clone(), other.clone(), x.clone()];
    Tensor::new(device, dtype, shape, Where, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn where_cond(&self, input: impl AsRef<Tensor>, other: impl AsRef<Tensor>) -> Tensor {
        where_cond(self, input.as_ref(), other.as_ref())
    }
}
