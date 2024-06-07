use crate::Tensor;
use std::{iter, rc::Rc};

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("incompatible shape lhs: {lhs:?}, rhs: {rhs:?}")]
    IncompatibleShape { lhs: Vec<usize>, rhs: Vec<usize> },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub enum RaiResult<T> {
    Ok(T),
    Err(Rc<Error>),
}

impl<T> From<RaiResult<T>> for Result<T> {
    fn from(result: RaiResult<T>) -> Self {
        match result {
            RaiResult::Ok(v) => Ok(v),
            RaiResult::Err(e) => Err((*e).clone()),
        }
    }
}

impl<T> From<Result<T>> for RaiResult<T> {
    fn from(result: Result<T>) -> Self {
        match result {
            Ok(v) => RaiResult::Ok(v),
            Err(e) => RaiResult::Err(Rc::new(e)),
        }
    }
}

impl<T> RaiResult<T> {
    pub fn to_std_result(self) -> std::result::Result<T, Error> {
        self.into()
    }

    pub fn as_ref(&self) -> RaiResult<&T> {
        match self {
            RaiResult::Ok(v) => RaiResult::Ok(v),
            RaiResult::Err(e) => RaiResult::Err(e.clone()),
        }
    }
}

#[macro_export]
macro_rules! try_get {
    ($v:expr) => {
        match $crate::RaiResult::from($v) {
            $crate::RaiResult::Ok(v) => v,
            $crate::RaiResult::Err(e) => return $crate::RaiResult::Err(e.clone()),
        }
    };
}

pub trait TryAsTensor {
    fn try_as_tensor(&self) -> RaiResult<&Tensor>;
}

impl TryAsTensor for Tensor {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        RaiResult::Ok(self)
    }
}

impl<'a> TryAsTensor for &'a Tensor {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        RaiResult::Ok(*self)
    }
}

impl TryAsTensor for RaiResult<Tensor> {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        match self {
            RaiResult::Ok(v) => RaiResult::Ok(v),
            RaiResult::Err(e) => RaiResult::Err(e.clone()),
        }
    }
}

impl<'a> TryAsTensor for RaiResult<&'a Tensor> {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        match self {
            RaiResult::Ok(v) => RaiResult::Ok(*v),
            RaiResult::Err(e) => RaiResult::Err(e.clone()),
        }
    }
}

impl<'a> TryAsTensor for &'a RaiResult<Tensor> {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        match self {
            RaiResult::Ok(v) => RaiResult::Ok(v),
            RaiResult::Err(e) => RaiResult::Err(e.clone()),
        }
    }
}

impl From<Tensor> for RaiResult<Tensor> {
    fn from(v: Tensor) -> Self {
        RaiResult::Ok(v)
    }
}
