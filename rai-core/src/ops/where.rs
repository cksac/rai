use crate::{Op, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Where;

impl Op for Where {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let pred = &primals[2];
        let tangent_t = &tangents[0];
        let tangent_f = &tangents[1];
        pred.where_cond(tangent_t, tangent_f)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let pred = &primals[2];
        let zeros = &cotangent.zeros_like();
        let contangent_t = pred.where_cond(cotangent, zeros);
        let contangent_f = pred.where_cond(zeros, cotangent);
        vec![contangent_t, contangent_f]
    }
}
