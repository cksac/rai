use crate::{
    ops::{Full, Normal, Random, ToDType},
    Op, Tensor,
};
use half::{bf16, f16};
use num_traits::Float;
use std::{
    any::Any,
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

pub trait ElemType: Clone + Copy + Debug + PartialEq + 'static {
    type DType: Type<Repr = Self>;
    fn zero() -> Self;
    fn one() -> Self;
    fn elem_count(start: Self, stop: Self, step: Self) -> usize;
    #[allow(unused_variables)]
    fn from_f64(val: f64) -> Option<Self> {
        None
    }
}

pub trait FloatElemType: ElemType + Add + Sub + Div + Mul {
    fn linspace(start: Self, end: Self, steps: usize) -> Vec<Self>;
}

pub trait Type: Clone + Copy + Debug + PartialEq + 'static {
    type Repr: ElemType<DType = Self>;

    fn boxed_dtype() -> Box<dyn DType>;

    fn zero() -> Self::Repr {
        Self::Repr::zero()
    }

    fn one() -> Self::Repr {
        Self::Repr::one()
    }

    fn from_f64(val: f64) -> Option<Self::Repr> {
        Self::Repr::from_f64(val)
    }

    fn primitive_full_zero() -> Full<Self> {
        Full::new(Self::Repr::zero())
    }

    fn primitive_full_one() -> Full<Self> {
        Full::new(Self::Repr::one())
    }

    fn primitive_rand() -> Random<Self> {
        Random::new(Self::Repr::zero(), Self::Repr::one())
    }

    fn primitive_randn() -> Normal<Self> {
        Normal::new(Self::Repr::zero(), Self::Repr::one())
    }

    fn primitive_as_type(&self) -> ToDType<Self> {
        ToDType::new(*self)
    }

    fn size_of_elem() -> usize {
        std::mem::size_of::<Self::Repr>()
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype;
}

pub trait DType: Debug {
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn DType>;
    fn equal(&self, rhs: &dyn DType) -> bool;
    fn primitive_full_zero(&self) -> Box<dyn Op>;
    fn primitive_full_one(&self) -> Box<dyn Op>;
    fn primitive_rand(&self) -> Box<dyn Op>;
    fn primitive_randn(&self) -> Box<dyn Op>;
    fn primitive_as_dtype(&self) -> Box<dyn Op>;
    fn size_of_elem(&self) -> usize;
    fn safetensor_dtype(&self) -> safetensors::Dtype;
}

impl<D: Type> DType for D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn DType> {
        Box::new(*self)
    }

    fn equal(&self, rhs: &dyn DType) -> bool {
        rhs.as_any()
            .downcast_ref()
            .map_or(false, |other| self == other)
    }

    fn primitive_full_zero(&self) -> Box<dyn Op> {
        Box::new(Self::primitive_full_zero())
    }

    fn primitive_full_one(&self) -> Box<dyn Op> {
        Box::new(Self::primitive_full_one())
    }

    fn primitive_rand(&self) -> Box<dyn Op> {
        Box::new(Self::primitive_rand())
    }

    fn primitive_randn(&self) -> Box<dyn Op> {
        Box::new(Self::primitive_randn())
    }

    fn primitive_as_dtype(&self) -> Box<dyn Op> {
        Box::new(self.primitive_as_type())
    }

    fn size_of_elem(&self) -> usize {
        Self::size_of_elem()
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        (*self).safetensor_dtype()
    }
}

impl<'a> PartialEq for &'a dyn DType {
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(*rhs)
    }
}

pub trait AsDType: Debug {
    fn dtype(&self) -> &dyn DType;
    fn into_boxed_dtype(self) -> Box<dyn DType>;
}

impl<D: DType> AsDType for D {
    fn dtype(&self) -> &dyn DType {
        self
    }

    fn into_boxed_dtype(self) -> Box<dyn DType> {
        self.clone_boxed()
    }
}

impl AsDType for Box<dyn DType> {
    fn dtype(&self) -> &dyn DType {
        self.as_ref()
    }

    fn into_boxed_dtype(self) -> Box<dyn DType> {
        self
    }
}

impl<'a> AsDType for &'a dyn DType {
    fn dtype(&self) -> &dyn DType {
        *self
    }

    fn into_boxed_dtype(self) -> Box<dyn DType> {
        DType::clone_boxed(self)
    }
}

impl AsDType for Tensor {
    fn dtype(&self) -> &dyn DType {
        Tensor::dtype(self)
    }

    fn into_boxed_dtype(self) -> Box<dyn DType> {
        Tensor::dtype(&self).clone_boxed()
    }
}

impl<'a> AsDType for &'a Tensor {
    fn dtype(&self) -> &dyn DType {
        Tensor::dtype(self)
    }

    fn into_boxed_dtype(self) -> Box<dyn DType> {
        Tensor::dtype(self).clone_boxed()
    }
}

impl ElemType for u8 {
    type DType = U8;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn elem_count(start: Self, stop: Self, step: Self) -> usize {
        (stop - start).div_ceil(step) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U8;
impl Type for U8 {
    type Repr = u8;

    fn boxed_dtype() -> Box<dyn DType> {
        Box::new(U8)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::U8
    }
}

impl ElemType for u32 {
    type DType = U32;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn elem_count(start: Self, stop: Self, step: Self) -> usize {
        (stop - start).div_ceil(step) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U32;
impl Type for U32 {
    type Repr = u32;

    fn boxed_dtype() -> Box<dyn DType> {
        Box::new(U32)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::U32
    }
}

impl ElemType for f16 {
    type DType = F16;

    fn zero() -> Self {
        f16::from(0i8)
    }

    fn one() -> Self {
        f16::from(1i8)
    }

    fn elem_count(start: Self, stop: Self, step: Self) -> usize {
        std::cmp::max(((stop - start) / step).ceil().to_f32() as usize, 0)
    }

    fn from_f64(val: f64) -> Option<Self> {
        Some(f16::from_f64(val))
    }
}

impl FloatElemType for f16 {
    fn linspace(start: Self, end: Self, steps: usize) -> Vec<Self> {
        match steps {
            0 => vec![],
            1 => vec![start],
            s => {
                let delta = (end - start).to_f64() / (s - 1) as f64;
                let mut data = (1..s - 1)
                    .map(|i| start + f16::from_f64(i as f64 * delta))
                    .collect::<Vec<_>>();
                data.insert(0, start);
                data.push(end);
                data
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F16;
impl Type for F16 {
    type Repr = f16;

    fn boxed_dtype() -> Box<dyn DType> {
        Box::new(F16)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F16
    }
}

impl ElemType for bf16 {
    type DType = BF16;

    fn zero() -> Self {
        bf16::from(0i8)
    }

    fn one() -> Self {
        bf16::from(1i8)
    }

    fn elem_count(start: Self, stop: Self, step: Self) -> usize {
        std::cmp::max(((stop - start) / step).ceil().to_f32() as usize, 0)
    }

    fn from_f64(val: f64) -> Option<Self> {
        Some(bf16::from_f64(val))
    }
}

impl FloatElemType for bf16 {
    fn linspace(start: Self, end: Self, steps: usize) -> Vec<Self> {
        match steps {
            0 => vec![],
            1 => vec![start],
            s => {
                let delta = (end - start).to_f64() / (s - 1) as f64;
                let mut data = (1..s - 1)
                    .map(|i| start + bf16::from_f64(i as f64 * delta))
                    .collect::<Vec<_>>();
                data.insert(0, start);
                data.push(end);
                data
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BF16;
impl Type for BF16 {
    type Repr = bf16;

    fn boxed_dtype() -> Box<dyn DType> {
        Box::new(BF16)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::BF16
    }
}

impl ElemType for f32 {
    type DType = F32;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn elem_count(start: Self, stop: Self, step: Self) -> usize {
        std::cmp::max(((stop - start) / step).ceil() as usize, 0)
    }

    fn from_f64(val: f64) -> Option<Self> {
        Some(val as f32)
    }
}

impl FloatElemType for f32 {
    fn linspace(start: Self, end: Self, steps: usize) -> Vec<Self> {
        match steps {
            0 => vec![],
            1 => vec![start],
            s => {
                let delta = ((end - start) as f64) / (s - 1) as f64;
                let mut data = (1..s - 1)
                    .map(|i| start + (i as f64 * delta) as f32)
                    .collect::<Vec<_>>();
                data.insert(0, start);
                data.push(end);
                data
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F32;
impl Type for F32 {
    type Repr = f32;

    fn boxed_dtype() -> Box<dyn DType> {
        Box::new(F32)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F32
    }
}

impl ElemType for f64 {
    type DType = F64;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn elem_count(start: Self, stop: Self, step: Self) -> usize {
        std::cmp::max(((stop - start) / step).ceil() as usize, 0)
    }

    fn from_f64(val: f64) -> Option<Self> {
        Some(val)
    }
}

impl FloatElemType for f64 {
    fn linspace(start: Self, end: Self, steps: usize) -> Vec<Self> {
        match steps {
            0 => vec![],
            1 => vec![start],
            s => {
                let delta = (end - start) / (s - 1) as f64;
                let mut data = (1..s - 1)
                    .map(|i| start + i as f64 * delta)
                    .collect::<Vec<_>>();
                data.insert(0, start);
                data.push(end);
                data
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F64;
impl Type for F64 {
    type Repr = f64;

    fn boxed_dtype() -> Box<dyn DType> {
        Box::new(F64)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::F64
    }
}

impl ElemType for i64 {
    type DType = I64;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn elem_count(start: Self, stop: Self, step: Self) -> usize {
        std::cmp::max(((stop - start) as f64 / step as f64).ceil() as usize, 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct I64;
impl Type for I64 {
    type Repr = i64;

    fn boxed_dtype() -> Box<dyn DType> {
        Box::new(I64)
    }

    fn safetensor_dtype(&self) -> safetensors::Dtype {
        safetensors::Dtype::I64
    }
}

impl From<safetensors::Dtype> for Box<dyn DType> {
    fn from(value: safetensors::Dtype) -> Self {
        match value {
            safetensors::Dtype::BOOL => todo!(),
            safetensors::Dtype::U8 => U8.into_boxed_dtype(),
            safetensors::Dtype::I8 => todo!(),
            safetensors::Dtype::I16 => todo!(),
            safetensors::Dtype::U16 => todo!(),
            safetensors::Dtype::F16 => F16.into_boxed_dtype(),
            safetensors::Dtype::BF16 => BF16.into_boxed_dtype(),
            safetensors::Dtype::I32 => todo!(),
            safetensors::Dtype::U32 => U32.into_boxed_dtype(),
            safetensors::Dtype::F32 => F32.into_boxed_dtype(),
            safetensors::Dtype::F64 => F64.into_boxed_dtype(),
            safetensors::Dtype::I64 => I64.into_boxed_dtype(),
            safetensors::Dtype::U64 => todo!(),
            _ => todo!(),
        }
    }
}

impl<'a> From<&'a dyn DType> for safetensors::Dtype {
    fn from(value: &'a dyn DType) -> Self {
        value.safetensor_dtype()
    }
}
