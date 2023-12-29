use std::any::Any;

use crate::{Primitive, Shape, Tensor};

#[derive(Debug, Clone)]
pub struct Broadcast {
    pub shape: Vec<usize>,
}
impl Broadcast {
    pub fn new(shape: impl Shape) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

impl Primitive for Broadcast {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Broadcast({:?})", self.shape)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        todo!()
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let shape = x.shape().to_vec();
        let diff = cotangent.ndim() - shape.ndim();
        let mut axes = Vec::new();
        for i in 0..cotangent.ndim() {
            if i < diff || shape[i - diff] != cotangent.shape_at(i) {
                axes.push(i);
            }
        }
        let cotangent_x = cotangent.reduce_sum((axes, true)).reshape(shape);
        vec![cotangent_x]
    }
}

#[derive(Debug, Clone)]
pub struct Reshape;
impl Primitive for Reshape {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        todo!()
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.reshape(x);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Transpose {
    pub axes: Vec<usize>,
}

impl Transpose {
    pub fn new(axes: impl Into<Vec<usize>>) -> Self {
        Self { axes: axes.into() }
    }
}

impl Primitive for Transpose {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Transpose({:?})", &self.axes)
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.transpose(&*self.axes)
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let mut axes = vec![0; self.axes.len()];
        for i in 0..self.axes.len() {
            axes[self.axes[i]] = i;
        }
        let cotangent_x = cotangent.transpose(axes);
        vec![cotangent_x]
    }
}
