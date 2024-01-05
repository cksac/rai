#![feature(fn_traits)]
#![feature(unboxed_closures)]

pub mod ops;
pub mod primitives;
pub use primitives::Primitive;

mod shape;
pub use shape::Shape;

mod tensor;
pub use tensor::Tensor;

mod dtype;
pub use dtype::{DType, ElemType};

pub mod backend;
pub use backend::Backend;

mod transforms;
pub use transforms::{eval, grad, jvp, value_and_grad, vjp, FromTensorMap, Func, WithTensors};

pub mod dispatch;

pub mod utils;

mod error;
pub use error::{Error, Result};

mod nn;
pub use nn::Module;
