pub mod ops;
pub mod primitives;

pub use primitives::Primitive;

mod shape;
pub use shape::{Dim, Dims, Shape};

mod tensor;
pub use tensor::Tensor;

mod dtype;
pub use dtype::{DType, DynDType, ElemType, F32, F64, U8};

pub mod backend;
pub use backend::Backend;

mod transforms;
pub use transforms::{
    eval, grad, jvp, raiexpr, value_and_grad, vjp, Aux, BasicType, Func, TensorIter,
    ValuAssociated, Value, VF,
};

pub mod dispatch;

pub mod utils;

mod error;
pub use error::{Error, Result};

mod nn;
pub use nn::{Module, ModuleType, NonTrainableModule, TrainableModule};

#[macro_use]
mod macros;
