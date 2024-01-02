use std::any::Any;

use crate::{Primitive, Tensor};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Negative;

impl Primitive for Negative {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        -tangent_x
    }

    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = -cotangent;
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sin;
impl Primitive for Sin {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        x.cos() * tangent_x
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * x.cos();
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Cos;
impl Primitive for Cos {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        -x.sin() * tangent_x
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * -x.sin();
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Square;
impl Primitive for Square {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        x * 2.0 * tangent_x
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * x * 2.0;
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sqrt;

impl Primitive for Sqrt {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        tangent_x * 0.5 / x.sqrt()
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * 0.5 / x.sqrt();
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Rsqrt;

impl Primitive for Rsqrt {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        -0.5 * tangent_x * (x.rsqrt() / x)
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = -0.5 * cotangent * (x.rsqrt() / x);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sign;

impl Primitive for Sign {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        x.zeros_like()
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = x.zeros_like();
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Abs;

impl Primitive for Abs {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let x = &primals[0];
        let tangent_x = &tangents[0];
        tangent_x * x.sign()
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent * x.sign();
        vec![cotangent_x]
    }
}
