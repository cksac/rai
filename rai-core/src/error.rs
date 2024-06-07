use crate::Tensor;

#[derive(thiserror::Error, Debug, Clone)]
pub enum Error {
    #[error("incompatible shape lhs: {lhs:?}, rhs: {rhs:?}")]
    IncompatibleShape { lhs: Vec<usize>, rhs: Vec<usize> },
}

pub type StdResult<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct RaiResult<T>(pub StdResult<T>);

impl<T> From<RaiResult<T>> for StdResult<T> {
    fn from(result: RaiResult<T>) -> Self {
        result.0
    }
}

impl<T> From<StdResult<T>> for RaiResult<T> {
    fn from(result: StdResult<T>) -> Self {
        RaiResult(result)
    }
}

impl<T> RaiResult<T> {
    pub fn from_val(val: T) -> Self {
        RaiResult(StdResult::Ok(val))
    }

    pub fn as_ref(&self) -> RaiResult<&T> {
        RaiResult(self.0.as_ref().map_err(|e| e.clone()))
    }
}

#[macro_export]
macro_rules! try_get {
    ($r:expr) => {
        match $r.0 {
            $crate::StdResult::Ok(v) => v,
            $crate::StdResult::Err(e) => {
                return $crate::RaiResult($crate::StdResult::Err(e.clone()))
            }
        }
    };
}

pub trait TryAsTensor {
    fn try_as_tensor(&self) -> RaiResult<&Tensor>;
}

impl TryAsTensor for Tensor {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        RaiResult(StdResult::Ok(self))
    }
}

impl<'a> TryAsTensor for &'a Tensor {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        RaiResult(StdResult::Ok(*self))
    }
}

impl TryAsTensor for RaiResult<Tensor> {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        self.as_ref()
    }
}

impl<'a> TryAsTensor for &'a RaiResult<Tensor> {
    fn try_as_tensor(&self) -> RaiResult<&Tensor> {
        self.as_ref()
    }
}

impl From<Tensor> for RaiResult<Tensor> {
    fn from(v: Tensor) -> Self {
        RaiResult(StdResult::Ok(v))
    }
}

impl<A, V: FromIterator<A>> FromIterator<RaiResult<A>> for RaiResult<V> {
    fn from_iter<I: IntoIterator<Item = RaiResult<A>>>(iter: I) -> Self {
        let mut vec = vec![];
        for v in iter {
            match v.0 {
                Ok(v) => vec.push(v),
                Err(e) => return RaiResult(Err(e)),
            }
        }
        RaiResult(Ok(vec.into_iter().collect()))
    }
}
