use std::collections::HashMap;

use rai::{
    nn::{self, gather_params, update_params, LayerNorm, Linear, Relu},
    trainable_module, Module, Tensor,
};

pub struct ModelConfig {
    pub num_vocab: usize,
    pub model_dim: usize,
}

pub struct RoPE {}

impl Module for RoPE {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {}
}

trainable_module!(RoPE);

pub struct PhiAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    rope: RoPE,
}

impl Module for PhiAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {}
}

trainable_module!(PhiAttention);

pub struct PhiMLP {
    fc1: Linear,
    fc2: Linear,
    act: Relu, // TODO: use GELU
}

impl Module for PhiMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {}
}

trainable_module!(PhiMLP);

pub struct PhiDecoderLayer {
    self_attn: PhiAttention,
    input_layernorm: LayerNorm,
    mlp: PhiMLP,
}

impl Module for PhiDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {}
}

trainable_module!(PhiDecoderLayer);

pub struct PhiModel {
    embed_tokens: nn::Embedding,
    layers: Vec<PhiDecoderLayer>,
    final_layernorm: LayerNorm,
}

impl Module for PhiModel {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
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

trainable_module!(PhiModel);

pub struct Model {
    model: PhiModel,
    lm_head: Linear,
}

impl Module for Model {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {}
}

trainable_module!(Model);

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
