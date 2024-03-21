use rai_core::Tensor;
use rai_derive::Module;

#[derive(Clone, Debug, Copy, Module)]
#[module(crate = rai_core, trainable = false, input = (Tensor, bool))]
pub struct Dropout {
    p: f32,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Self { p }
    }

    pub fn fwd(&self, x: &Tensor, train: bool) -> Tensor {
        if train {
            x.dropout(self.p)
        } else {
            x.clone()
        }
    }
}
