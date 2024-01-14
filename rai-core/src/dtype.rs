use std::{any::Any, fmt::Debug};

use candle_core::Tensor;

use crate::{
    ops,
    primitives::{AsType, Full},
    Backend, Primitive, Shape,
};

pub trait ElemType: Clone + Debug + 'static {
    type DType: DType<Repr = Self>;
    fn dtype() -> Self::DType;
    fn dyn_dtype() -> Box<dyn DynDType> {
        Box::from(Self::dtype())
    }
}

pub trait DType: Clone + Copy + Debug + PartialEq + 'static {
    type Repr: ElemType<DType = Self>;
    fn zero(&self) -> Self::Repr;
    fn one(&self) -> Self::Repr;

    fn full_zero(&self) -> Full<Self> {
        Full::new(self.zero())
    }

    fn full_one(&self) -> Full<Self> {
        Full::new(self.one())
    }

    fn as_self_dtype(&self) -> AsType<Self> {
        AsType::new(*self)
    }
}

pub trait DynDType: Debug {
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn DynDType>;
    fn equal(&self, rhs: &dyn DynDType) -> bool;
    fn full_zero(&self) -> Box<dyn Primitive>;
    fn full_one(&self) -> Box<dyn Primitive>;
    fn as_self_dtype(&self) -> Box<dyn Primitive>;
}

impl<'a> PartialEq for &'a dyn DynDType {
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(*rhs)
    }
}

// impl<'a> Debug for &'a dyn DType {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         todo!()
//     }
// }

impl<T> From<T> for Box<dyn DynDType>
where
    T: DType + 'static,
{
    fn from(value: T) -> Self {
        Box::new(value)
    }
}

impl<'a> From<&'a dyn DynDType> for Box<dyn DynDType> {
    fn from(t: &'a dyn DynDType) -> Self {
        t.clone_boxed()
    }
}

// impl<T> From<T> for Box<dyn DType>
// where
//     T: Clone + DTypeRepr + 'static,
// {
//     fn from(t: T) -> Self {
//         Box::new(t.clone())
//     }
// }

impl<D: DType> DynDType for D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn DynDType> {
        Box::new(self.clone())
    }

    fn equal(&self, rhs: &dyn DynDType) -> bool {
        rhs.as_any()
            .downcast_ref()
            .map_or(false, |other| self == other)
    }

    fn full_zero(&self) -> Box<dyn Primitive> {
        Box::new(self.full_zero())
    }

    fn full_one(&self) -> Box<dyn Primitive> {
        Box::new(self.full_one())
    }

    fn as_self_dtype(&self) -> Box<dyn Primitive> {
        Box::new(self.as_self_dtype())
    }
}

impl ElemType for u8 {
    type DType = U8;

    fn dtype() -> Self::DType {
        U8
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U8;
impl DType for U8 {
    type Repr = u8;

    fn zero(&self) -> Self::Repr {
        0
    }

    fn one(&self) -> Self::Repr {
        1
    }
}

impl ElemType for f32 {
    type DType = F32;

    fn dtype() -> Self::DType {
        F32
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F32;
impl DType for F32 {
    type Repr = f32;

    fn zero(&self) -> Self::Repr {
        0.0
    }

    fn one(&self) -> Self::Repr {
        1.0
    }
}

impl ElemType for f64 {
    type DType = F64;

    fn dtype() -> Self::DType {
        F64
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F64;
impl DType for F64 {
    type Repr = f64;

    fn zero(&self) -> Self::Repr {
        0.0
    }

    fn one(&self) -> Self::Repr {
        1.0
    }
}
