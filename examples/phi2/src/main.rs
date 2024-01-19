use std::collections::{btree_map::Keys, HashMap};

use rai::{
    nn::{self, gather_params, update_params, LayerNorm, Linear, Module, Relu},
    trainable_module, Shape, Tensor,
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

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }
}

trainable_module!(RoPE);

pub struct PhiAttention {
    num_heads: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    repeats: usize,

    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    rope: RoPE,
}

impl Module for PhiAttention {
    type Input = (Tensor, Option<Tensor>, Option<Tensor>); // (x, mask, cache)
    type Output = (Tensor, Tensor, Tensor); // (output, keys, values)

    fn forward(&self, x: &Self::Input) -> Self::Output {
        let queries = self.q_proj.forward(&x.0);
        let keys = self.k_proj.forward(&x.0);
        let values = self.v_proj.forward(&x.0);

        // Extract some shapes
        let [B, L, D]: [usize; 3] = queries.shape_of([0, 1, 2]).try_into().unwrap();

        // Prepare the queries, keys and values for the attention computation
        let queries = queries
            .reshape(&[B, L, self.num_heads, self.head_dim])
            .transpose([0, 2, 1, 3]);

        let Keys = keys
            .reshape([B, L, self.num_key_value_heads, self.head_dim])
            .transpose([0, 2, 1, 3]);

        let values = values
            .reshape([B, L, self.num_key_value_heads, self.head_dim])
            .transpose([0, 2, 1, 3]);

        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }
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

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }
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

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }
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
        gather_params!(params, @self.embed_tokens);
        gather_params!(params, []self.layers);
        gather_params!(params, @self.final_layernorm);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        update_params!(params, @self.embed_tokens);
        update_params!(params, []self.layers);
        update_params!(params, @self.final_layernorm);
    }

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }
}

trainable_module!(PhiModel);

pub struct Model {
    model: PhiModel,
    lm_head: Linear,
}

impl Module for Model {
    type Input = (Tensor, Tensor, Tensor); // (input, mask, cache)
    type Output = (Tensor, Tensor); // (output, cache)

    fn forward(&self, x: &Self::Input) -> Self::Output {
        todo!()
    }

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {}

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        todo!()
    }
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
