use crate::Tensor;
use rustc_hash::FxHashMap;
use std::collections::hash_map::{Entry, Keys, Values};

#[derive(Debug, Clone, Default)]
pub struct TensorMap {
    tensors: FxHashMap<usize, Tensor>,
}

impl TensorMap {
    pub fn new() -> Self {
        Self {
            tensors: FxHashMap::default(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tensors: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn insert(&mut self, id: usize, grad: Tensor) {
        self.tensors.insert(id, grad);
    }

    pub fn get(&self, id: usize) -> Option<&Tensor> {
        self.tensors.get(&id)
    }

    pub fn entry(&mut self, id: usize) -> Entry<usize, Tensor> {
        self.tensors.entry(id)
    }

    pub fn remove(&mut self, id: usize) -> Option<Tensor> {
        self.tensors.remove(&id)
    }

    pub fn keys(&self) -> Keys<'_, usize, Tensor> {
        self.tensors.keys()
    }

    pub fn values(&self) -> Values<'_, usize, Tensor> {
        self.tensors.values()
    }

    pub fn clear(&mut self) {
        self.tensors.clear();
    }
}

impl FromIterator<(usize, Tensor)> for TensorMap {
    fn from_iter<T: IntoIterator<Item = (usize, Tensor)>>(iter: T) -> Self {
        Self {
            tensors: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for TensorMap {
    type Item = (usize, Tensor);
    type IntoIter = std::collections::hash_map::IntoIter<usize, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.tensors.into_iter()
    }
}
