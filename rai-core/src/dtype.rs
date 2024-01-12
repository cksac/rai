use std::{fmt::Debug, hash::Hash};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    F32,
    F64,
}

pub trait ElemType: Debug + Clone + PartialEq + Send + Sync + 'static {
    const DTYPE: DType;
    fn one() -> Self;
    fn zero() -> Self;
    fn into_f64(self) -> f64;
}

impl ElemType for u8 {
    const DTYPE: DType = DType::U8;

    fn one() -> Self {
        1
    }
    fn zero() -> Self {
        0
    }
    fn into_f64(self) -> f64 {
        self as f64
    }
}

impl ElemType for f32 {
    const DTYPE: DType = DType::F32;

    fn one() -> Self {
        1.0f32
    }
    fn zero() -> Self {
        0.0f32
    }
    fn into_f64(self) -> f64 {
        self as f64
    }
}

impl ElemType for f64 {
    const DTYPE: DType = DType::F64;

    fn one() -> Self {
        1.0f64
    }
    fn zero() -> Self {
        0.0f64
    }
    fn into_f64(self) -> f64 {
        self
    }
}
