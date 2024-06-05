use crate::Tensor;
use rustc_hash::FxHashMap;
use std::collections::hash_map::{Entry, Keys, Values};

#[derive(Debug, Clone, Default)]
pub struct GradMap {
    grads: FxHashMap<usize, Tensor>,
}

impl GradMap {
    pub fn new() -> Self {
        Self {
            grads: FxHashMap::default(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            grads: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    pub fn len(&self) -> usize {
        self.grads.len()
    }

    pub fn insert(&mut self, id: usize, grad: Tensor) {
        self.grads.insert(id, grad);
    }

    pub fn get(&self, id: usize) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    pub fn entry(&mut self, id: usize) -> Entry<usize, Tensor> {
        self.grads.entry(id)
    }

    pub fn remove(&mut self, id: usize) -> Option<Tensor> {
        self.grads.remove(&id)
    }

    pub fn keys(&self) -> Keys<'_, usize, Tensor> {
        self.grads.keys()
    }

    pub fn values(&self) -> Values<'_, usize, Tensor> {
        self.grads.values()
    }

    pub fn clear(&mut self) {
        self.grads.clear();
    }
}

impl FromIterator<(usize, Tensor)> for GradMap {
    fn from_iter<T: IntoIterator<Item = (usize, Tensor)>>(iter: T) -> Self {
        Self {
            grads: iter.into_iter().collect(),
        }
    }
}
