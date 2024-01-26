use crate::Tensor;
use std::{any::Any, fmt::Debug};

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

impl<'a> From<&'a dyn Primitive> for Box<dyn Primitive> {
    fn from(t: &'a dyn Primitive) -> Self {
        t.clone_boxed()
    }
}

mod creation;
pub use creation::*;

mod binary;
pub use binary::*;

mod unary;
pub use unary::*;

mod transform;
pub use transform::*;

mod reduce;
pub use reduce::*;

mod indexing;
pub use indexing::*;
