use crate::{Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMul;

impl Op for MatMul {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        tangents[0].matmul(&primals[1]) + primals[0].matmul(&tangents[1])
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let cotangent_lhs = cotangent.matmul(rhs.t());
        let cotangent_rhs = lhs.t().matmul(cotangent);
        vec![cotangent_lhs, cotangent_rhs]
    }
}

#[track_caller]
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let device = lhs.device();
    let dtype = lhs.dtype();
    let lhs_in = lhs;
    let rhs_in = rhs;
    let mut lhs = lhs.clone();
    let mut rhs = rhs.clone();
    if lhs.rank() == 1 {
        lhs = lhs.reshape([&[1], lhs.shape()].concat());
    }
    if rhs.rank() == 1 {
        rhs = rhs.reshape([rhs.shape(), &[1]].concat());
    }
    let (mut shape, lhs_shape, rhs_shape, lhs_b, rhs_b) = lhs
        .shape_broadcast_matmul(&rhs)
        .unwrap_or_else(|e| panic!("matmul({:?}, {:?}) with error {:?}", lhs, rhs, e));
    let inputs = match (lhs_b, rhs_b) {
        (false, false) => vec![lhs.clone(), rhs.clone()],
        (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&rhs_shape)],
        (true, false) => vec![lhs.broadcast_to_unchecked(&lhs_shape), rhs.clone()],
        (true, true) => vec![
            lhs.broadcast_to_unchecked(&lhs_shape),
            rhs.broadcast_to_unchecked(&rhs_shape),
        ],
    };
    if lhs_in.rank() == 1 || rhs_in.rank() == 1 {
        let erase_start = shape.len() - if lhs_in.rank() == 1 { 2 } else { 1 };
        let erase_end = shape.len() - if rhs_in.rank() == 1 { 0 } else { 1 };
        let matml_out = Tensor::new(device, dtype, &shape, MatMul, inputs);
        shape.drain(erase_start..erase_end);
        matml_out.reshape(shape)
    } else {
        Tensor::new(device, dtype, shape, MatMul, inputs)
    }
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn matmul<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        matmul(self, rhs.as_ref())
    }
}
