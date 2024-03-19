use rai_core::{AsDevice, Shape, Tensor, Type};
use rai_derive::Module;

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub fn new(
        num_embeddings: usize,
        features: usize,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let weight = Tensor::rand([num_embeddings, features], dtype, device);
        Self { weight }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let mut out_dims = x.shape().to_vec();
        out_dims.push(self.weight.shape_at(-1));
        let index = &x.flatten(..);
        self.weight.index_select(0, index).reshape(out_dims)
    }
}
