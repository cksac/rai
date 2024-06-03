use crate::{Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct PowerFloat {
    pub exponent: f64,
}

impl PowerFloat {
    pub fn new(exponent: f64) -> Self {
        Self { exponent }
    }

    pub fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl Op for PowerFloat {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("PowerFloat({:?})", &self.exponent)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        tangent_x * x.powf(self.exponent - 1.0) * self.exponent
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * x.powf(self.exponent - 1.0) * self.exponent;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn powf(x: &Tensor, exponent: f64) -> Tensor {
    let device = x.device();
    let dtype = x.dtype(); // todo: promote to f64?
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, PowerFloat::new(exponent), inputs)
}
