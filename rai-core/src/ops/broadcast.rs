use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Broadcast {
    pub shape: Vec<usize>,
}
impl Broadcast {
    pub fn new(shape: impl Shape) -> Self {
        Self {
            shape: shape.shape().to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Op for Broadcast {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Broadcast")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Broadcast({:?})", self.shape)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.broadcast_to(self.shape())
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let shape = x.shape().to_vec();
        let diff = cotangent.rank() - shape.rank();
        let mut dims = Vec::new();
        for i in 0..cotangent.rank() {
            if i < diff || shape[i - diff] != cotangent.size(i) {
                dims.push(i);
            }
        }
        let cotangent_x = cotangent.sum((dims, true)).reshape(&shape);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn broadcast_to(x: &Tensor, shape: impl Shape) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let inputs = vec![x.clone()];
    match x.shape_broadcast_to(&shape) {
        Ok((out_shape, _, _)) => {
            Tensor::new(device, dtype, out_shape, Broadcast::new(shape), inputs)
        }
        Err(e) => {
            let shape = [];
            Tensor::err(device, dtype, shape, Broadcast::new(shape), inputs, e)
        }
    }
}

#[track_caller]
pub fn broadcast_to_unchecked(x: &Tensor, shape: impl Shape) -> Tensor {
    let device = x.device();
    let dtype = x.dtype();
    let inputs = vec![x.clone()];
    Tensor::new(device, dtype, &shape, Broadcast::new(&shape), inputs)
}

#[track_caller]
pub fn broadcast_left(x: &Tensor, shape: impl Shape) -> Tensor {
    let out_shape = x.shape_expand_left(&shape);
    x.broadcast_to_unchecked(out_shape)
}

#[track_caller]
pub fn broadcast_right(x: &Tensor, shape: impl Shape) -> Tensor {
    let out_shape = x.shape_expand_right(&shape);
    let mut x = x.clone();
    for _ in x.rank()..out_shape.rank() {
        x = x.unsqueeze(-1);
    }
    x.broadcast_to_unchecked(out_shape)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn broadcast_to(&self, shape: impl Shape) -> Tensor {
        broadcast_to(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn broadcast_to_unchecked(&self, shape: impl Shape) -> Tensor {
        broadcast_to_unchecked(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn broadcast_left(&self, shape: impl Shape) -> Tensor {
        broadcast_left(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn broadcast_right(&self, shape: impl Shape) -> Tensor {
        broadcast_right(self, shape)
    }
}
