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

    #[error("param not found: {0}")]
    ParamNotFound(String),

    #[error("no data")]
    NoData,

    #[error("downcast error from: {from}, to: {to}")]
    Downcast {
        from: &'static str,
        to: &'static str,
    },

    #[error("missing weight map")]
    MissingWeightMap,

    #[error("invalid weight map: {0}")]
    InvalidWeightMap(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),

    #[error(transparent)]
    HfHub(#[from] hf_hub::api::sync::ApiError),
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum OpError {
    #[error("incompatible shape lhs: {lhs:?}, rhs: {rhs:?}")]
    IncompatibleShape { lhs: Vec<usize>, rhs: Vec<usize> },
}

pub type OpResult<T> = std::result::Result<T, OpError>;
