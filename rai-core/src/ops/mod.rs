use crate::{Dim, Dims, Shape, Tensor};
use std::{any::Any, fmt::Debug};

mod full;
pub use full::*;

mod normal;
pub use normal::*;

mod random;
pub use random::*;

mod arange;
pub use arange::*;

mod from_array;
pub use from_array::*;

mod concatenate;
pub use concatenate::*;

mod add;
pub use add::*;

mod sub;
pub use sub::*;

mod mul;
pub use mul::*;

mod div;
pub use div::*;

mod matmul;
pub use matmul::*;

mod equal;
pub use equal::*;

mod not_equal;
pub use not_equal::*;

mod greater;
pub use greater::*;

mod greater_equal;
pub use greater_equal::*;

mod less;
pub use less::*;

mod less_equal;
pub use less_equal::*;

mod maximum;
pub use maximum::*;

mod minimum;
pub use minimum::*;

mod negative;
pub use negative::*;

mod sin;
pub use sin::*;

mod cos;
pub use cos::*;

mod square;
pub use square::*;

mod sqrt;
pub use sqrt::*;

mod rsqrt;
pub use rsqrt::*;

mod sign;
pub use sign::*;

mod abs;
pub use abs::*;

mod exp;
pub use exp::*;

mod log;
pub use log::*;

mod log2;
pub use log2::*;

mod log10;
pub use log10::*;

mod softmax;
pub use softmax::*;

mod log_softmax;
pub use log_softmax::*;

mod erf;
pub use erf::*;

mod tanh;
pub use tanh::*;

mod power_float;
pub use power_float::*;

mod broadcast;
pub use broadcast::*;

mod reshape;
pub use reshape::*;

mod transpose;
pub use transpose::*;

mod to_contiguous;
pub use to_contiguous::*;

mod to_device;
pub use to_device::*;

mod to_dtype;
pub use to_dtype::*;

mod permute;
pub use permute::*;

mod max_pool1d;
pub use max_pool1d::*;

mod max_pool2d;
pub use max_pool2d::*;

mod avg_pool1d;
pub use avg_pool1d::*;

mod avg_pool2d;
pub use avg_pool2d::*;

mod reduce_sum;
pub use reduce_sum::*;

mod reduce_max;
pub use reduce_max::*;

mod reduce_min;
pub use reduce_min::*;

mod argmax;
pub use argmax::*;

mod argmin;
pub use argmin::*;

mod gather;
pub use gather::*;

mod index_add;
pub use index_add::*;

mod scatter_add;
pub use scatter_add::*;

mod index_select;
pub use index_select::*;

mod narrow;
pub use narrow::*;

mod r#where;
pub use r#where::*;

mod conv1d;
pub use conv1d::*;

mod conv2d;
pub use conv2d::*;

mod conv_transpose1d;
pub use conv_transpose1d::*;

mod conv_transpose2d;
pub use conv_transpose2d::*;

mod upsample_nearest1d;
pub use upsample_nearest1d::*;

mod upsample_nearest2d;
pub use upsample_nearest2d::*;

mod flash_attention;
pub use flash_attention::*;

#[macro_export]
macro_rules! impl_std_ops_for_scalar {
    ($T:ty, $op:ident, $func:ident) => {
        impl std::ops::$op<Tensor> for $T {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: Tensor) -> Self::Output {
                let lhs = rhs.full_like::<$T>(self);
                $func(&lhs, &rhs)
            }
        }

        impl<'a> std::ops::$op<&'a Tensor> for $T {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: &'a Tensor) -> Self::Output {
                let lhs = rhs.full_like::<$T>(self);
                $func(&lhs, &rhs)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_std_ops {
    ($op:ident, $func:ident) => {
        impl std::ops::$op<Tensor> for Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(&self, &rhs)
            }
        }

        impl<'a> std::ops::$op<&'a Tensor> for Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: &'a Tensor) -> Tensor {
                $func(&self, rhs)
            }
        }

        impl<'a> std::ops::$op<Tensor> for &'a Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(self, &rhs)
            }
        }

        impl<'a, 'b> std::ops::$op<&'b Tensor> for &'a Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: &'b Tensor) -> Tensor {
                $func(self, rhs)
            }
        }

        impl<T> std::ops::$op<T> for Tensor
        where
            T: $crate::ElemType,
        {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like::<T>(rhs);
                $func(&self, &rhs)
            }
        }

        impl<'a, T> std::ops::$op<T> for &'a Tensor
        where
            T: $crate::ElemType,
        {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like::<T>(rhs);
                $func(self, &rhs)
            }
        }

        $crate::impl_std_ops_for_scalar!(f32, $op, $func);
        $crate::impl_std_ops_for_scalar!(f64, $op, $func);
        $crate::impl_std_ops_for_scalar!(u8, $op, $func);
    };
}

#[macro_export]
macro_rules! broadcast_binary_op {
    ($(#[$meta:meta])* $primitive:ident, $func:ident) => {
        $(#[$meta])*
        #[track_caller]
        pub fn $func(lhs: &Tensor, rhs: &Tensor) -> Tensor {
            let device = lhs.device();
            let dtype = lhs.dtype();
            let (shape, lhs_b, rhs_b) = lhs.shape_broadcast_to(rhs).unwrap_or_else(|e| {
                panic!(
                    "{}({:?}, {:?}) with error {:?}",
                    stringify!($func),
                    lhs,
                    rhs,
                    e
                )
            });
            let inputs = match (lhs_b, rhs_b) {
                (false, false) => vec![lhs.clone(), rhs.clone()],
                (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&shape)],
                (true, false) => vec![lhs.broadcast_to_unchecked(&shape), rhs.clone()],
                (true, true) => vec![lhs.broadcast_to_unchecked(&shape), rhs.broadcast_to_unchecked(&shape)],
            };
            Tensor::new(device, dtype, shape, $primitive, inputs)
        }
    };
    ($(#[$meta:meta])* $primitive:ident, $func:ident, $out_ty:ident) => {
        $(#[$meta])*
        #[track_caller]
        pub fn $func(lhs: &Tensor, rhs: &Tensor) -> Tensor {
            let device = lhs.device();
            let dtype = $out_ty;
            let (shape, lhs_b, rhs_b) = lhs.shape_broadcast_to(rhs).unwrap_or_else(|e| {
                panic!(
                    "{}({:?}, {:?}) with error {:?}",
                    stringify!($func),
                    lhs,
                    rhs,
                    e
                )
            });
            let inputs = match (lhs_b, rhs_b) {
                (false, false) => vec![lhs.clone(), rhs.clone()],
                (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&shape)],
                (true, false) => vec![lhs.broadcast_to_unchecked(&shape), rhs.clone()],
                (true, true) => vec![lhs.broadcast_to_unchecked(&shape), rhs.broadcast_to_unchecked(&shape)],
            };
            Tensor::new(device, dtype, shape, $primitive, inputs)
        }
    };
}

pub trait ReduceArgs: Debug {
    fn dims(&self) -> &impl Dims<Vec<usize>>;
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> ReduceArgs for T
where
    T: Dims<Vec<usize>>,
{
    fn dims(&self) -> &impl Dims<Vec<usize>> {
        self
    }
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> ReduceArgs for (T, bool)
where
    T: Dims<Vec<usize>>,
{
    fn dims(&self) -> &impl Dims<Vec<usize>> {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

impl<T> ReduceArgs for (T, usize)
where
    T: Dims<Vec<usize>>,
{
    fn dims(&self) -> &impl Dims<Vec<usize>> {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> ReduceArgs for (T, bool, usize)
where
    T: Dims<Vec<usize>>,
{
    fn dims(&self) -> &impl Dims<Vec<usize>> {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

pub trait ArgReduceArgs: Debug {
    fn dim(&self) -> &impl Dim;
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T: Dim> ArgReduceArgs for T {
    fn dim(&self) -> &impl Dim {
        self
    }
}

impl<T: Dim> ArgReduceArgs for (T, bool) {
    fn dim(&self) -> &impl Dim {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

pub trait ToPair<T>: Debug {
    fn to_pair(&self) -> (T, T);
}

impl ToPair<usize> for usize {
    fn to_pair(&self) -> (usize, usize) {
        (*self, *self)
    }
}

impl ToPair<usize> for [usize; 2] {
    fn to_pair(&self) -> (usize, usize) {
        (self[0], self[1])
    }
}

impl ToPair<usize> for (usize, usize) {
    fn to_pair(&self) -> (usize, usize) {
        *self
    }
}

pub trait Op: Debug {
    fn clone_boxed(&self) -> Box<dyn Op>;
    fn dot_label(&self) -> String {
        format!("{:?}", self)
    }
    fn as_any(&self) -> &dyn Any;
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor;
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor>;
}

impl<T> From<T> for Box<dyn Op>
where
    T: Clone + Op + 'static,
{
    fn from(t: T) -> Self {
        Box::new(t.clone())
    }
}

impl Clone for Box<dyn Op> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl<'a> From<&'a dyn Op> for Box<dyn Op> {
    fn from(t: &'a dyn Op) -> Self {
        t.clone_boxed()
    }
}

fn reduce_chooser_jvp_rule(g: &Tensor, ans: &Tensor, operand: &Tensor, dims: &[usize]) -> Tensor {
    let mut shape = operand.shape().to_vec();
    for dim in dims {
        shape[*dim] = 1;
    }
    let location_indicators = operand.eq(ans.reshape(shape)).to_dtype(g);
    let counts = location_indicators.sum(dims);
    (g * location_indicators).sum(dims) / counts
}
