use std::any::Any;

use crate::{Primitive, Shape, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Add;

impl Primitive for Add {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs + tangent_rhs
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_lhs = cotangent.clone();
        let cotangent_rhs = cotangent.clone();
        vec![cotangent_lhs, cotangent_rhs]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sub;

impl Primitive for Sub {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs - tangent_rhs
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_lhs = cotangent.clone();
        let cotangent_rhs = -cotangent;
        vec![cotangent_lhs, cotangent_rhs]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mul;

impl Primitive for Mul {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        tangent_lhs * rhs + tangent_rhs * lhs
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let cotangent_lhs = cotangent * rhs;
        let cotangent_rhs = cotangent * lhs;
        vec![cotangent_lhs, cotangent_rhs]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Div;

impl Primitive for Div {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let tangent_lhs = &tangents[0];
        let tangent_rhs = &tangents[1];
        (tangent_lhs - lhs * tangent_rhs) / rhs
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let cotangent_lhs = cotangent * rhs;
        let cotangent_rhs = cotangent * -lhs / rhs;
        vec![cotangent_lhs, cotangent_rhs]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatMul;

impl Primitive for MatMul {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        tangents[0].matmul(&primals[1]) + primals[0].matmul(&tangents[1])
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let lhs = &primals[0];
        let rhs = &primals[1];
        let ndim = cotangent.ndim();
        let mut axes = (0..ndim).collect::<Vec<usize>>();
        // swap last two axes
        axes.swap(ndim - 1, ndim - 2);
        let axes = axes.as_slice();
        let cotangent_lhs = cotangent.matmul(rhs.transpose(axes));
        let cotangent_rhs = lhs.transpose(axes).matmul(cotangent);
        vec![cotangent_lhs, cotangent_rhs]
    }
}
