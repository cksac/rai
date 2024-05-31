use crate::{Primitive, Tensor};
use std::{any::Any, f64::consts::PI};
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Erf;

impl Primitive for Erf {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        (2. / PI.sqrt()) * (x.square().neg()).exp() * tangent_x
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = (2. / PI.sqrt()) * (x.square().neg()).exp() * cotangent;
        vec![cotangent_x]
    }
}
