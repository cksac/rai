use crate::Tensor;
use rustc_hash::FxHashMap;
use std::collections::hash_map::{Entry, IntoIter, Iter, Keys, Values};

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

    #[inline]
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    #[inline]
    pub fn insert(&mut self, id: usize, grad: Tensor) {
        self.grads.insert(id, grad);
    }

    #[inline]
    pub fn get(&self, id: usize) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    #[inline]
    pub fn entry(&mut self, id: usize) -> Entry<usize, Tensor> {
        self.grads.entry(id)
    }

    #[inline]
    pub fn remove(&mut self, id: usize) -> Option<Tensor> {
        self.grads.remove(&id)
    }

    #[inline]
    pub fn keys(&self) -> Keys<'_, usize, Tensor> {
        self.grads.keys()
    }

    #[inline]
    pub fn values(&self) -> Values<'_, usize, Tensor> {
        self.grads.values()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.grads.clear();
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, usize, Tensor> {
        self.grads.iter()
    }
}

impl FromIterator<(usize, Tensor)> for GradMap {
    fn from_iter<T: IntoIterator<Item = (usize, Tensor)>>(iter: T) -> Self {
        Self {
            grads: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for GradMap {
    type Item = (usize, Tensor);
    type IntoIter = IntoIter<usize, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.grads.into_iter()
    }
}
