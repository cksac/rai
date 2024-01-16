use core::fmt::Debug;
use rai_core::{Backend, DType, Module, Tensor};

pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    pub fn new(
        num_embeddings: usize,
        features: usize,
        dtype: impl DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Self {
        let backend = &backend.into();
        // TODO: init strategy
        let weight = Tensor::normal([num_embeddings, features], dtype, backend);
        Self { weight }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        // TODO:
        input.clone()
    }
}
