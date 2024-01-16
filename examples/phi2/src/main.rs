use std::{collections::HashMap, path::Path};

use rai::{
    backend::Cpu,
    differentiable_module,
    nn::{self, gather_params, update_params, LayerNorm, Linear, Relu},
    Module, Tensor, F32,
};

pub struct ModelConfig {
    pub num_vocab: usize,
    pub model_dim: usize,
}

pub struct RoPE {}
impl Module for RoPE {
    fn forward(&self, x: &Tensor) -> Tensor {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }
}
differentiable_module!(RoPE);

pub struct PhiAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    rope: RoPE,
}
impl Module for PhiAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }
}
differentiable_module!(PhiAttention);

pub struct PhiMLP {
    fc1: Linear,
    fc2: Linear,
    act: Relu, // TODO: use GELU
}
impl Module for PhiMLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }
}
differentiable_module!(PhiMLP);

pub struct PhiDecoderLayer {
    self_attn: PhiAttention,
    input_layernorm: LayerNorm,
    mlp: PhiMLP,
}
impl Module for PhiDecoderLayer {
    fn forward(&self, x: &Tensor) -> Tensor {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        todo!()
    }
}
differentiable_module!(PhiDecoderLayer);

pub struct PhiModel {
    embed_tokens: nn::Embedding,
    layers: Vec<PhiDecoderLayer>,
    final_layernorm: LayerNorm,
}
impl Module for PhiModel {
    fn forward(&self, x: &Tensor) -> Tensor {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(@self.embed_tokens, params);
        gather_params!([]self.layers, params);
        gather_params!(@self.final_layernorm, params);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(@self.embed_tokens, params);
        update_params!([]self.layers, params);
        update_params!(@self.final_layernorm, params);
    }
}
differentiable_module!(PhiModel);

pub struct Model {
    model: PhiModel,
    lm_head: Linear,
}
impl Module for Model {
    fn forward(&self, x: &Tensor) -> Tensor {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        gather_params!(@self.model, params);
        gather_params!(@self.lm_head, params);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(@self.model, params);
        update_params!(@self.lm_head, params);
    }
}

// fn load_model(config: ModelConfig, model_path: impl AsRef<Path>) -> PhiModel {
//     // todo: Load model weights from file
//     let mut weights: HashMap<usize, Tensor> = HashMap::new();
//     let model = PhiModel::new(config);
//     // TODO: update model weights using weight's name instead of tensor id?
//     model.update_params(&mut weights);
//     model
// }

fn main() {
    // let model_path = "./microsoft/phi-2";
    // let config = ModelConfig {
    //     num_vocab: 50257,
    //     model_dim: 768,
    // };
    // let model = load_model(config, model_path);
}
