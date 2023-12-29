use std::any::Any;

use crate::{Primitive, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Full {
    pub val: f64,
}
impl Full {
    pub fn new(val: f64) -> Self {
        Full { val }
    }
}

impl Primitive for Full {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Full({})", self.val)
    }

    fn as_any(&self) -> &dyn Any {
        self
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Normal;

impl Primitive for Normal {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
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

#[derive(Clone, Debug, PartialEq)]
pub struct Arange {
    pub start: f64,
    pub stop: f64,
    pub step: f64,
}

impl Arange {
    pub fn new(start: f64, stop: f64, step: f64) -> Self {
        Self { start, stop, step }
    }
}

impl Primitive for Arange {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Arange({}, {}, {})", self.start, self.stop, self.step)
    }

    fn as_any(&self) -> &dyn Any {
        self
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
