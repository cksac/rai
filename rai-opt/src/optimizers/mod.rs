use rai_core::Tensor;
use std::collections::BTreeMap;

pub trait Optimizer {
    /// return new value for the parameters
    fn step(
        &mut self,
        params: Vec<Tensor>,
        grads: &BTreeMap<usize, Tensor>,
    ) -> BTreeMap<usize, Tensor>;
}

mod sdg;
pub use sdg::SDG;
