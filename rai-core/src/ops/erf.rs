use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow, f64::consts::PI};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Erf;

impl Op for Erf {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Erf")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        (2. / PI.sqrt()) * (x.square().neg()).exp() * tangent_x
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = (2. / PI.sqrt()) * (x.square().neg()).exp() * cotangent;
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn erf(x: &Tensor) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, shape, Erf, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn erf(&self) -> Tensor {
        erf(self)
    }
}
