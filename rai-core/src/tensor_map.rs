use crate::Tensor;
use rustc_hash::FxHashMap;
use std::collections::hash_map::{Entry, IntoIter, Iter, Keys, Values};

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

    #[inline]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    #[inline]
    pub fn insert(&mut self, id: usize, grad: Tensor) {
        self.tensors.insert(id, grad);
    }

    #[inline]
    pub fn get(&self, id: usize) -> Option<&Tensor> {
        self.tensors.get(&id)
    }

    #[inline]
    pub fn entry(&mut self, id: usize) -> Entry<usize, Tensor> {
        self.tensors.entry(id)
    }

    #[inline]
    pub fn remove(&mut self, id: usize) -> Option<Tensor> {
        self.tensors.remove(&id)
    }

    #[inline]
    pub fn keys(&self) -> Keys<'_, usize, Tensor> {
        self.tensors.keys()
    }

    #[inline]
    pub fn values(&self) -> Values<'_, usize, Tensor> {
        self.tensors.values()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.tensors.clear();
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, usize, Tensor> {
        self.tensors.iter()
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
    type IntoIter = IntoIter<usize, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.tensors.into_iter()
    }
}
