use std::borrow::Cow;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("op error {0}")]
    Op(#[from] OpError),

    #[error("unimplemented op: {op:?}, device: {device:?}")]
    Unimplemented {
        op: Cow<'static, str>,
        device: Cow<'static, str>,
    },
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum OpError {
    #[error("incompatible shape lhs: {lhs:?}, rhs: {rhs:?}")]
    IncompatibleShape { lhs: Vec<usize>, rhs: Vec<usize> },
}

pub type OpResult<T> = std::result::Result<T, OpError>;
