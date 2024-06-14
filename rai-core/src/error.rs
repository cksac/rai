use crate::{Device, Op};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("incompatible shape lhs: {lhs:?}, rhs: {rhs:?}")]
    IncompatibleShape { lhs: Vec<usize>, rhs: Vec<usize> },

    #[error("unimplemented op: {op:?}, device: {device:?}")]
    Unimplemented {
        op: Box<dyn Op>,
        device: Box<dyn Device>,
    },
}
