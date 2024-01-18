use std::collections::HashMap;

use rai_core::{non_trainable_module, Tensor};

macro_rules! impl_activation {
    ($M:ty, $OP:tt) => {
        impl rai_core::nn::Module for $M {
            type Input = Tensor;
            type Output = Tensor;

            fn forward(&self, x: &Self::Input) -> Self::Output {
                rai_core::ops::$OP(x)
            }
            fn gather_params(&self, _: &mut HashMap<usize, Tensor>) {}
            fn update_params(&self, _: &mut HashMap<usize, Tensor>) {}
        }

        non_trainable_module!($M);
    };
}

#[derive(Clone, Debug, Copy)]
pub struct Relu;
impl_activation!(Relu, relu);

// TODO: GELU
#[derive(Clone, Debug, Copy)]
pub struct Gelu;
impl_activation!(Gelu, relu);
