use std::collections::HashMap;

use rai_core::{non_differentiable, Module, Tensor};

macro_rules! impl_activation {
    ($M:ty, $OP:tt) => {
        impl Module for $M {
            type Input<'i> = &'i Tensor;
            type Output<'o> = Tensor;
            fn forward<'i, 'o>(&self, x: Self::Input<'i>) -> Self::Output<'o> {
                rai_core::ops::$OP(x)
            }
            fn gather_params(&self, _: &mut HashMap<usize, Tensor>) {}
            fn update_params(&self, _: &mut HashMap<usize, Tensor>) {}
        }
    };
}

#[derive(Clone, Debug, Copy)]
pub struct Relu;
impl_activation!(Relu, relu);
non_differentiable!(Relu);

// TODO: GELU
#[derive(Clone, Debug, Copy)]
pub struct Gelu;
impl_activation!(Gelu, relu);
non_differentiable!(Gelu);
