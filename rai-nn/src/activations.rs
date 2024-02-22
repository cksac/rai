use rai_core::Tensor;
use rai_derive::Module;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Default, Module)]
#[module(crate = rai_core)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Silu,
}

impl Activation {
    pub fn fwd(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Relu => x.relu(),
            Activation::Gelu => x.gelu(),
            Activation::NewGelu => x.new_gelu(),
            Activation::Silu => x.silu(),
        }
    }
}

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core)]
pub struct Relu;
impl Relu {
    pub fn fwd(&self, x: &Tensor) -> Tensor {
        x.relu()
    }
}

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core)]
pub struct Gelu;
impl Gelu {
    pub fn fwd(&self, x: &Tensor) -> Tensor {
        x.gelu()
    }
}

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core)]
pub struct NewGelu;
impl NewGelu {
    pub fn fwd(&self, x: &Tensor) -> Tensor {
        x.new_gelu()
    }
}

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core)]
pub struct Silu;
impl Silu {
    pub fn fwd(&self, x: &Tensor) -> Tensor {
        x.silu()
    }
}
