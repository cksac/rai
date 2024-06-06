use crate::{AsDevice, ElemType, Op, Tensor, Type, F16, F32, F64, U32, U8};
use half::f16;
use std::any::Any;
use std::fmt::Debug;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct Arange<D: Type> {
    pub start: D::Repr,
    pub stop: D::Repr,
    pub step: D::Repr,
}

impl<D: Type> Arange<D> {
    pub fn new(start: D::Repr, stop: D::Repr, step: D::Repr) -> Self {
        Self { start, stop, step }
    }
}

impl<D: Type> Op for Arange<D> {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Arange({:?}, {:?}, {:?})", self.start, self.stop, self.step)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    #[inline]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], _tangents: &[Tensor]) -> Tensor {
        output.ones_like()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    #[inline]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], _cotangent: &Tensor) -> Vec<Tensor> {
        vec![]
    }
}

macro_rules! impl_arange_args {
    ($R:ty, $T:tt) => {
        impl From<$R> for ArangeArgs<$T> {
            fn from(stop: $R) -> Self {
                Self::new(<$R as ElemType>::zero(), stop, <$R as ElemType>::one())
            }
        }

        impl From<($R, $R)> for ArangeArgs<$T> {
            fn from(args: ($R, $R)) -> Self {
                Self::new(args.0, args.1, <$R as ElemType>::one())
            }
        }

        impl From<($R, $R, $R)> for ArangeArgs<$T> {
            fn from(args: ($R, $R, $R)) -> Self {
                Self::new(args.0, args.1, args.2)
            }
        }
    };
}

impl_arange_args!(f32, F32);
impl_arange_args!(f64, F64);
impl_arange_args!(f16, F16);
impl_arange_args!(u8, U8);
impl_arange_args!(u32, U32);

pub struct ArangeArgs<D: Type> {
    pub start: D::Repr,
    pub stop: D::Repr,
    pub step: D::Repr,
}
impl<D: Type> ArangeArgs<D> {
    pub fn new(start: D::Repr, stop: D::Repr, step: D::Repr) -> Self {
        Self { start, stop, step }
    }

    pub fn elem_count(&self) -> usize {
        D::Repr::elem_count(self.start, self.stop, self.step)
    }
}

/// Creates a 1-D `Tensor` with values from a range.
///
/// # Arguments
///
/// * `args` - The arguments for the `arange` function.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A 1-D `Tensor` with values from the specified range.
#[track_caller]
pub fn arange<D: Type, A: Into<ArangeArgs<D>>>(args: A, device: impl AsDevice) -> Tensor {
    let args = args.into();
    let start = args.start;
    let stop = args.stop;
    let step = args.step;
    let size = args.elem_count();
    let dtype = D::boxed_dtype();
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        [size],
        Arange::<D>::new(start, stop, step),
        inputs,
    )
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn arange<D: Type, A: Into<ArangeArgs<D>>>(args: A, device: impl AsDevice) -> Tensor {
        arange(args, device)
    }
}
