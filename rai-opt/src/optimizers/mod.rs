use rai_core::{GradMap, TensorMap};

pub trait Optimizer {
    /// return new value for the parameters
    fn step(&mut self, grads: &GradMap) -> TensorMap;
}

mod sdg;
pub use sdg::SDG;
