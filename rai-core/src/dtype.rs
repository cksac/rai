use std::{any::Any, fmt::Debug};

use half::f16;

use crate::{
    primitives::{AsType, Full},
    Primitive,
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
    fn zero() -> Self::Repr;
    fn one() -> Self::Repr;

    fn primitive_full_zero() -> Full<Self> {
        Full::new(Self::zero())
    }

    fn primitive_full_one() -> Full<Self> {
        Full::new(Self::one())
    }

    fn primitive_as_type(&self) -> AsType<Self> {
        AsType::new(*self)
    }

    fn size_of_elem() -> usize {
        std::mem::size_of::<Self::Repr>()
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype;
}

pub trait DynDType: Debug {
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn DynDType>;
    fn equal(&self, rhs: &dyn DynDType) -> bool;
    fn primitive_full_zero(&self) -> Box<dyn Primitive>;
    fn primitive_full_one(&self) -> Box<dyn Primitive>;
    fn primitive_as_dtype(&self) -> Box<dyn Primitive>;
    fn size_of_elem(&self) -> usize;
    fn safetensor_dtype(&self) -> safetensors::Dtype;
}

impl<'a> PartialEq for &'a dyn DynDType {
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(*rhs)
    }
}

impl<T> From<T> for Box<dyn DynDType>
where
    T: DType,
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

impl<D: DType> DynDType for D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn DynDType> {
        Box::new(*self)
    }

    fn equal(&self, rhs: &dyn DynDType) -> bool {
        rhs.as_any()
            .downcast_ref()
            .map_or(false, |other| self == other)
    }

    fn primitive_full_zero(&self) -> Box<dyn Primitive> {
        Box::new(Self::primitive_full_zero())
    }

    fn primitive_full_one(&self) -> Box<dyn Primitive> {
        Box::new(Self::primitive_full_one())
    }

    fn primitive_as_dtype(&self) -> Box<dyn Primitive> {
        Box::new(self.primitive_as_type())
    }

    fn size_of_elem(&self) -> usize {
        Self::size_of_elem()
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        (*self).safetensor_dtype()
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

    fn zero() -> Self::Repr {
        0
    }

    fn one() -> Self::Repr {
        1
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::U8
    }
}

impl ElemType for u32 {
    type DType = U32;

    fn dtype() -> Self::DType {
        U32
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U32;
impl DType for U32 {
    type Repr = u32;

    fn zero() -> Self::Repr {
        0
    }

    fn one() -> Self::Repr {
        1
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::U32
    }
}

impl ElemType for f16 {
    type DType = F16;

    fn dtype() -> Self::DType {
        F16
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F16;
impl DType for F16 {
    type Repr = f16;

    fn zero() -> Self::Repr {
        f16::from(0i8)
    }

    fn one() -> Self::Repr {
        f16::from(1i8)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F16
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

    fn zero() -> Self::Repr {
        0.0
    }

    fn one() -> Self::Repr {
        1.0
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F32
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

    fn zero() -> Self::Repr {
        0.0
    }

    fn one() -> Self::Repr {
        1.0
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F64
    }
}

impl From<safetensors::Dtype> for Box<dyn DynDType> {
    fn from(value: safetensors::Dtype) -> Self {
        match value {
            safetensors::Dtype::BOOL => todo!(),
            safetensors::Dtype::U8 => U8.into(),
            safetensors::Dtype::I8 => todo!(),
            safetensors::Dtype::I16 => todo!(),
            safetensors::Dtype::U16 => todo!(),
            safetensors::Dtype::F16 => F16.into(),
            safetensors::Dtype::BF16 => todo!(),
            safetensors::Dtype::I32 => todo!(),
            safetensors::Dtype::U32 => todo!(),
            safetensors::Dtype::F32 => F32.into(),
            safetensors::Dtype::F64 => F64.into(),
            safetensors::Dtype::I64 => todo!(),
            safetensors::Dtype::U64 => todo!(),
            _ => todo!(),
        }
    }
}

impl<'a> From<&'a dyn DynDType> for safetensors::Dtype {
    fn from(value: &'a dyn DynDType) -> Self {
        value.safetensor_dtype()
    }
}
