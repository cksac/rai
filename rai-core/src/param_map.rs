use crate::Tensor;
use rustc_hash::FxHashMap;
use std::collections::hash_map::{Entry, Keys, Values};

#[derive(Debug, Clone, Default)]
pub struct ParamMap {
    params: FxHashMap<String, Tensor>,
}

impl ParamMap {
    pub fn new() -> Self {
        Self {
            params: FxHashMap::default(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            params: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn insert(&mut self, id: String, grad: Tensor) {
        self.params.insert(id, grad);
    }

    pub fn get(&self, id: &str) -> Option<&Tensor> {
        self.params.get(id)
    }

    pub fn entry(&mut self, id: String) -> Entry<String, Tensor> {
        self.params.entry(id)
    }

    pub fn remove(&mut self, id: &str) -> Option<Tensor> {
        self.params.remove(id)
    }

    pub fn keys(&self) -> Keys<'_, String, Tensor> {
        self.params.keys()
    }

    pub fn values(&self) -> Values<'_, String, Tensor> {
        self.params.values()
    }

    pub fn clear(&mut self) {
        self.params.clear();
    }
}

impl FromIterator<(String, Tensor)> for ParamMap {
    fn from_iter<T: IntoIterator<Item = (String, Tensor)>>(iter: T) -> Self {
        Self {
            params: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for ParamMap {
    type Item = (String, Tensor);
    type IntoIter = std::collections::hash_map::IntoIter<String, Tensor>;

    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}
