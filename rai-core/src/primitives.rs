use crate::{Shape, Tensor};
use std::any::Any;
use std::fmt::Debug;
use std::vec;

pub trait Primitive: Debug {
    fn clone_boxed(&self) -> Box<dyn Primitive>;
    fn dot_label(&self) -> String {
        format!("{:?}", self)
    }
    fn as_any(&self) -> &dyn Any;
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor;
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor>;
}

impl<T> From<T> for Box<dyn Primitive>
where
    T: Clone + Primitive + 'static,
{
    fn from(t: T) -> Self {
        Box::new(t.clone())
    }
}

impl Clone for Box<dyn Primitive> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl From<&dyn Primitive> for Box<dyn Primitive> {
    fn from(t: &dyn Primitive) -> Self {
        t.clone_boxed()
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
pub struct Add;

impl Primitive for Add {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sub;

impl Primitive for Sub {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mul;

impl Primitive for Mul {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Div;

impl Primitive for Div {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
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
pub struct ReduceSum {
    pub axes: Vec<usize>,
}
impl ReduceSum {
    pub fn new(axes: impl Into<Vec<usize>>) -> Self {
        Self { axes: axes.into() }
    }
}

impl Primitive for ReduceSum {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("ReduceSum({:?})", &self.axes)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.reduce_sum((&self.axes, false))
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let mut shape = x.shape().to_vec();
        for axis in &self.axes {
            shape[*axis] = 1;
        }
        let cotangent_x = cotangent.reshape(shape).broadcast_to(x);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MatMul;

impl Primitive for MatMul {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
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
