use rai_core::Tensor;
use rai_derive::Module;

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core)]
pub struct Relu;
impl Relu {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        x.relu()
    }
}

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core)]
pub struct Gelu;
impl Gelu {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        x.gelu()
    }
}

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core)]
pub struct NewGelu;
impl NewGelu {
    pub fn apply(&self, x: &Tensor) -> Tensor {
        x.new_gelu()
    }
}
