use std::{collections::HashMap, path::Path};

use rai::{nn, Module, Tensor};

pub struct Phi2 {
    wte: nn::Embedding,
}

pub struct ModelConfig {
    pub num_vocab: usize,
    pub model_dim: usize,
}

impl Phi2 {
    pub fn new(config: ModelConfig) -> Self {
        let backend = &Cpu;
        let dtype = F32;

        let model = Phi2 {
            wte: nn::Embedding::new(config.num_vocab, config.model_dim, dtype, backend),
        };
        model
    }
}

impl nn::Module for Phi2 {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.wte.forward(input);
        x
    }
}

fn load_model(config: ModelConfig, model_path: impl AsRef<Path>) -> Phi2 {
    // todo: Load model weights from file
    let mut weights: HashMap<usize, Tensor> = HashMap::new();
    let model = Phi2::new(config);
    // TODO: update model weights using weight's name instead of tensor id?
    model.update(&mut weights);
    model
}

fn main() {
    let model_path = "./microsoft/phi-2";
    let config = ModelConfig {
        num_vocab: 50257,
        model_dim: 768,
    };
    let model = load_model(config, model_path);
}
