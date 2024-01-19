use rai_core::{non_trainable_module, Tensor};
use std::collections::HashMap;

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
            fn gather_named_params(&self, _: &str, _: &mut HashMap<String, Tensor>) {}
            fn update_named_params(&self, _: &str, _: &mut HashMap<String, Tensor>) {}
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
