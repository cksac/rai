use crate::{Shape, Tensor};
use std::{any::Any, fmt::Debug};

pub trait Primitive: Debug {
    fn clone_boxed(&self) -> Box<dyn Primitive>;
    fn dot_label(&self) -> String {
        format!("{:?}", self)
    }
    fn as_any(&self) -> &dyn Any;
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor;
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor>;
}

impl<T> From<T> for Box<dyn Primitive>
where
    T: Clone + Primitive + 'static,
{
    fn from(t: T) -> Self {
        Box::new(t.clone())
    }
}

impl Clone for Box<dyn Primitive> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl<'a> From<&'a dyn Primitive> for Box<dyn Primitive> {
    fn from(t: &'a dyn Primitive) -> Self {
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
