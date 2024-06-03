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

/// Represents the arguments for the `arange` function.
pub trait ArangeArgs<D: Type>: Debug {
    /// Returns the start value for the `arange` function.
    fn start(&self) -> D::Repr {
        D::Repr::zero()
    }

    /// Returns the stop value for the `arange` function.
    fn stop(&self) -> D::Repr;

    /// Returns the step value for the `arange` function.
    fn step(&self) -> D::Repr {
        D::Repr::one()
    }

    /// Returns the size of the resulting `Tensor` for the `arange` function.
    fn size(&self) -> usize {
        D::Repr::elem_count(self.start(), self.stop(), self.step())
    }
}

macro_rules! impl_arange_args {
    ($R:ty, $T:tt) => {
        impl ArangeArgs<$T> for $R {
            fn stop(&self) -> $R {
                *self
            }
        }

        impl ArangeArgs<$T> for ($R, $R) {
            fn start(&self) -> $R {
                self.0
            }

            fn stop(&self) -> $R {
                self.1
            }
        }

        impl ArangeArgs<$T> for ($R, $R, $R) {
            fn start(&self) -> $R {
                self.0
            }

            fn stop(&self) -> $R {
                self.1
            }

            fn step(&self) -> $R {
                self.2
            }
        }
    };
}

impl_arange_args!(f32, F32);
impl_arange_args!(f64, F64);
impl_arange_args!(f16, F16);
impl_arange_args!(u8, U8);
impl_arange_args!(u32, U32);

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
pub fn arange<D: Type, T: ArangeArgs<D>>(args: T, device: impl AsDevice) -> Tensor {
    let start = args.start();
    let stop = args.stop();
    let step = args.step();
    let dtype = D::boxed_dtype();
    let size = args.size();
    let inputs = vec![];
    Tensor::new(
        device,
        dtype,
        [size],
        Arange::<D>::new(start, stop, step),
        inputs,
    )
}
