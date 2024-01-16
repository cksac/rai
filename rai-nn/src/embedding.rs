use core::fmt::Debug;
use rai_core::{differentiable_module, Backend, DType, Module, Shape, Tensor};
use std::collections::HashMap;

use crate::{gather_params, update_params};

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

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut out_dims = input.shape().to_vec();
        out_dims.push(self.weight.shape_at(..));
        let index = &input.flatten(..);
        self.weight.index_select(0, index).reshape(out_dims)
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(self.weight, params);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(self.weight, params);
    }
}

differentiable_module!(Embedding);
