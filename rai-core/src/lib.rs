pub mod ops;
pub mod primitives;

pub use primitives::Primitive;

mod shape;
pub use shape::{Dim, Dims, Shape};

mod tensor;
pub use tensor::Tensor;

mod dtype;
pub use dtype::{DType, ElemType};

pub mod backend;
pub use backend::Backend;

mod transforms;
pub use transforms::{
    eval, grad, jvp, raiexpr, value_and_grad, vjp, Aux, Differentiable, Func, TensorIter,
};

pub mod dispatch;

pub mod utils;

mod error;
pub use error::{Error, Result};

mod nn;
pub use nn::{DifferentiableModule, Module};

#[macro_use]
mod macros;
