use std::collections::HashMap;

use rai_core::Tensor;

pub trait Optimizer {
    /// return new value for the parameters
    fn step(&mut self, grads: &HashMap<usize, Tensor>) -> HashMap<usize, Tensor>;
}

mod sdg;
pub use sdg::SDG;
