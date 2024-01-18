use core::fmt::Debug;
use rai_core::{
    Backend, DType, Module, ModuleType, Shape, Tensor, TrainableModule, ValuAssociated,
};
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

impl ValuAssociated for Embedding {
    type ValueType = ModuleType;
    type Tensors = HashMap<usize, Tensor>;
    type Gradient = HashMap<usize, Tensor>;
}

impl Module for Embedding {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        let mut out_dims = x.shape().to_vec();
        out_dims.push(self.weight.shape_at(..));
        let index = &x.flatten(..);
        self.weight.index_select(0, index).reshape(out_dims)
    }
}

impl TrainableModule for Embedding {
    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(self.weight, params);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(self.weight, params);
    }
}
