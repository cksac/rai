use crate::Tensor;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

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

    pub fn clear(&mut self) {
        self.grads.clear();
    }
}
