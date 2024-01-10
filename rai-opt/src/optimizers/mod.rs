use rai_core::Tensor;
use std::collections::HashMap;

pub trait Optimizer {
    /// return new value for the parameters
    fn step(&mut self, grads: &HashMap<usize, Tensor>) -> HashMap<usize, Tensor>;
}

mod sdg;
pub use sdg::SDG;
