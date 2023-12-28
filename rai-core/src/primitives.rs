use crate::{Shape, Tensor};
use dyn_clone::DynClone;
use std::any::Any;
use std::fmt::Debug;
use std::vec;

pub trait Primitive: Debug + DynClone {
    fn dot_label(&self) -> String {
        format!("{:?}", self)
    }
    fn as_any(&self) -> &dyn Any;
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor;
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor>;
}

dyn_clone::clone_trait_object!(Primitive);

impl From<&Box<dyn Primitive>> for Box<dyn Primitive> {
    fn from(value: &Box<dyn Primitive>) -> Self {
        value.clone()
    }
}

impl<T> From<T> for Box<dyn Primitive>
where
    T: Primitive + 'static,
{
    fn from(t: T) -> Self {
        Box::new(t)
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
