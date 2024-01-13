use std::{any::Any, fmt::Debug};

use candle_core::Tensor;

use crate::{
    ops,
    primitives::{AsType, Full},
    Backend, Primitive, Shape,
};

pub trait ElemType: Clone + Debug + 'static {
    fn dtype() -> Box<dyn DType>;
}

pub trait DTypeRepr: Clone + Copy + Debug + PartialEq + 'static {
    type Repr: ElemType;
    fn zero(&self) -> Self::Repr;
    fn one(&self) -> Self::Repr;

    fn full_zero(&self) -> Full<Self::Repr> {
        Full::new(self.zero())
    }

    fn full_one(&self) -> Full<Self::Repr> {
        Full::new(self.one())
    }

    fn as_self_dtype(&self) -> AsType<Self> {
        AsType::new(*self)
    }
}

pub trait DType: Debug {
    // fn zero(&self) -> Box<dyn ElemType>;
    // fn one(&self) -> Box<dyn ElemType>;
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn DType>;
    fn equal(&self, rhs: &dyn DType) -> bool;
    fn full_zero(&self) -> Box<dyn Primitive>;
    fn full_one(&self) -> Box<dyn Primitive>;
    fn as_self_dtype(&self) -> Box<dyn Primitive>;
}

impl<'a> PartialEq for &'a dyn DType {
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(*rhs)
    }
}

// impl<'a> Debug for &'a dyn DType {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         todo!()
//     }
// }

impl<T> From<T> for Box<dyn DType>
where
    T: DTypeRepr + 'static,
{
    fn from(value: T) -> Self {
        Box::new(value)
    }
}

impl<'a> From<&'a dyn DType> for Box<dyn DType> {
    fn from(t: &'a dyn DType) -> Self {
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

impl<D: DTypeRepr> DType for D {
    // fn zero(&self) -> Box<dyn ElemType> {
    //     Box::new(self.zero())
    // }

    // fn one(&self) -> Box<dyn ElemType> {
    //     Box::new(self.one())
    // }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn DType> {
        Box::new(self.clone())
    }

    fn equal(&self, rhs: &dyn DType) -> bool {
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
    fn dtype() -> Box<dyn DType> {
        Box::new(U8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U8;
impl DTypeRepr for U8 {
    type Repr = u8;

    fn zero(&self) -> Self::Repr {
        0
    }

    fn one(&self) -> Self::Repr {
        1
    }
}

impl ElemType for f32 {
    fn dtype() -> Box<dyn DType> {
        Box::new(F32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F32;
impl DTypeRepr for F32 {
    type Repr = f32;

    fn zero(&self) -> Self::Repr {
        0.0
    }

    fn one(&self) -> Self::Repr {
        1.0
    }
}

impl ElemType for f64 {
    fn dtype() -> Box<dyn DType> {
        Box::new(F64)
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F64;
impl DTypeRepr for F64 {
    type Repr = f64;

    fn zero(&self) -> Self::Repr {
        0.0
    }

    fn one(&self) -> Self::Repr {
        1.0
    }
}
