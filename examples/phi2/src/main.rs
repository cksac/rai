use hf_hub::{api::sync::Api, Repo, RepoType};
use rai::{
    ext,
    nn::{
        self, gather_params, update_params, Embedding, LayerNorm, Linear, Module, NamePath, NewGelu,
    },
    trainable_module,
    utils::cuda_enabled,
    AsDevice, Cpu, Cuda, Device, Shape, Tensor, Type, F32,
};
use serde::Deserialize;
use std::{cell::RefCell, collections::HashMap, fmt::Debug, io::Write, time::Instant};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub partial_rotary_factor: f64,
    pub qk_layernorm: bool,
}

impl Config {
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    dim: usize,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let dim = (cfg.partial_rotary_factor * cfg.head_dim() as f64) as usize;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_array(inv_freq, [1, inv_freq_len], device).to_dtype(dtype);
        let t = Tensor::arange((0u32, cfg.max_position_embeddings as u32), device)
            .to_dtype(dtype)
            .reshape([cfg.max_position_embeddings, 1]);
        let freqs = t.matmul(&inv_freq);
        let emb = Tensor::cat(&[&freqs, &freqs], -1);
        Self {
            dim,
            sin: emb.sin(),
            cos: emb.cos(),
        }
    }
}

impl Module for RotaryEmbedding {
    type Input = (Tensor, usize);
    type Output = Tensor;

    /// input = (x, seqlen_offset)
    fn forward(&self, input: &Self::Input) -> Self::Output {
        let xs = &input.0;
        let seqlen_offset = input.1;
        let [_b_size, _num_heads, seq_len, headdim]: [usize; 4] =
            xs.shape_of([0, 1, 2, 3]).try_into().unwrap();
        let xs_rot = xs.narrow(3, 0, self.dim);
        let xs_pass = xs.narrow(3, self.dim, headdim - self.dim);
        let xs12 = xs_rot.chunk(2, -1);
        let (xs1, xs2) = (&xs12[0], &xs12[1]);
        let c = &self.cos.narrow(0, seqlen_offset, seq_len).to_dtype(xs);
        let s = &self.sin.narrow(0, seqlen_offset, seq_len).to_dtype(xs);
        let rotate_half = Tensor::cat(&[&xs2.neg(), xs1], -1);
        let xs_rot = xs_rot * c + rotate_half * s;
        Tensor::cat(&[xs_rot, xs_pass], -1)
    }

    fn gather_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn gather_named_params(&self, _prefix: &str, _params: &mut HashMap<String, Tensor>) {}

    fn update_named_params(&self, _prefix: &str, _params: &mut HashMap<String, Tensor>) {}
}

pub struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: NewGelu,
}

impl MLP {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let fc1 = Linear::new(cfg.hidden_size, cfg.intermediate_size, true, dtype, device);
        let fc2 = Linear::new(cfg.intermediate_size, cfg.hidden_size, true, dtype, device);
        Self {
            fc1,
            fc2,
            act: NewGelu,
        }
    }
}

impl Module for MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        (&self.fc1).chain(&self.act).chain(&self.fc2).forward(x)
    }

    fn gather_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn gather_named_params(&self, _prefix: &str, _params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.fc1.update_named_params(&prefix.push("fc1"), params);
        self.fc2.update_named_params(&prefix.push("fc2"), params);
    }
}

trainable_module!(MLP);

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    rotary_emb: RotaryEmbedding,
    softmax_scale: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: RefCell<Option<(Tensor, Tensor)>>,
}

fn get_mask(size: usize, device: impl AsDevice) -> Tensor {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_array(mask, [size, size], device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Tensor {
    let shape = mask.shape();
    let on_true = &Tensor::full(on_true, shape, on_false.device())
        .broadcast_to(shape)
        .to_dtype(on_false);
    mask.where_cond(on_true, on_false)
}

impl Attention {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();
        let q_proj = Linear::new(cfg.hidden_size, num_heads * head_dim, true, dtype, device);
        let k_proj = Linear::new(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            true,
            dtype,
            device,
        );
        let v_proj = Linear::new(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            true,
            dtype,
            device,
        );
        let dense = Linear::new(num_heads * head_dim, cfg.hidden_size, true, dtype, device);
        // Alternative rope scaling are not supported.
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, device);
        let (q_layernorm, k_layernorm) = if cfg.qk_layernorm {
            let q_layernorm = LayerNorm::new(head_dim, cfg.layer_norm_eps, true, dtype, device);
            let k_layernorm = LayerNorm::new(head_dim, cfg.layer_norm_eps, true, dtype, device);
            (Some(q_layernorm), Some(k_layernorm))
        } else {
            (None, None)
        };
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            q_layernorm,
            k_layernorm,
            rotary_emb,
            softmax_scale,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: RefCell::new(None),
        }
    }

    fn repeat_kv(&self, xs: Tensor) -> Tensor {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            xs
        } else {
            let [b_sz, num_kv_heads, seq_len, head_dim]: [usize; 4] =
                xs.shape_of([0, 1, 2, 3]).try_into().unwrap();
            xs.unsqueeze(2)
                .broadcast_to([b_sz, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([b_sz, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }
}

impl Module for Attention {
    type Input = (Tensor, Option<Tensor>); // (x, mask)
    type Output = Tensor; // output

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let xs = &input.0;
        let mask = input.1.as_ref();

        let [b_size, seq_len, _n_embd]: [usize; 3] = xs.shape_of([0, 1, 2]).try_into().unwrap();
        let query_states = self.q_proj.forward(xs);
        let key_states = self.k_proj.forward(xs);
        let value_states = self.v_proj.forward(xs);
        let query_states = match &self.q_layernorm {
            None => query_states,
            Some(ln) => ln.forward(&query_states),
        };
        let key_states = match &self.k_layernorm {
            None => key_states,
            Some(ln) => ln.forward(&key_states),
        };
        let query_states = query_states
            .reshape([b_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let key_states = key_states
            .reshape([b_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let value_states = value_states
            .reshape([b_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        // Rotary embeddings.
        let kv_cache = self.kv_cache.borrow();
        let seqlen_offset = match &*kv_cache {
            None => 0,
            Some((prev_k, _)) => prev_k.shape_at(2),
        };
        let query_states = self.rotary_emb.forward(&(query_states, seqlen_offset));
        let key_states = self.rotary_emb.forward(&(key_states, seqlen_offset));
        // KV cache.
        let (key_states, value_states) = match &*kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key_states], 2);
                let v = Tensor::cat(&[prev_v, &value_states], 2);
                (k, v)
            }
        };
        drop(kv_cache);
        self.kv_cache
            .replace(Some((key_states.clone(), value_states.clone())));
        // Repeat kv.
        let key_states = self.repeat_kv(key_states).to_contiguous();
        let value_states = self.repeat_kv(value_states).to_contiguous();

        // Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        let attn_weights = query_states
            .to_dtype(F32)
            .to_contiguous()
            .matmul(key_states.to_dtype(F32).t() * self.softmax_scale);
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => masked_fill(
                &attn_weights,
                &mask.broadcast_to(mask.shape_expand_left(&[b_size, self.num_heads])),
                f32::NEG_INFINITY,
            ),
        };
        let attn_weights = attn_weights.softmax(-1).to_dtype(&value_states);
        let attn_output = attn_weights.matmul(&value_states).transpose(1, 2);
        let attn_output = attn_output.reshape([b_size, seq_len, attn_output.size_of_dims(2..)]);
        self.dense.forward(&attn_output)
    }

    fn gather_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn gather_named_params(&self, _prefix: &str, _params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.q_proj
            .update_named_params(&prefix.push("q_proj"), params);
        self.k_proj
            .update_named_params(&prefix.push("k_proj"), params);
        self.v_proj
            .update_named_params(&prefix.push("v_proj"), params);
        self.dense
            .update_named_params(&prefix.push("dense"), params);
        if let Some(q_layernorm) = &self.q_layernorm {
            q_layernorm.update_named_params(&prefix.push("q_layernorm"), params);
        }
        if let Some(k_layernorm) = &self.k_layernorm {
            k_layernorm.update_named_params(&prefix.push("k_layernorm"), params);
        }
    }
}

trainable_module!(Attention);

pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
}

impl DecoderLayer {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let self_attn = Attention::new(cfg, dtype, device);
        let mlp = MLP::new(cfg, dtype, device);
        let input_layernorm =
            LayerNorm::new(cfg.hidden_size, cfg.layer_norm_eps, true, dtype, device);
        Self {
            self_attn,
            mlp,
            input_layernorm,
        }
    }
}

impl Module for DecoderLayer {
    type Input = (Tensor, Option<Tensor>); // (x, mask)
    type Output = Tensor; // output

    fn forward(&self, input: &Self::Input) -> Self::Output {
        let xs = &input.0;
        let mask = input.1.clone();
        let residual = xs;
        let xs = self.input_layernorm.forward(xs);
        let attn_outputs = self.self_attn.forward(&(xs.clone(), mask));
        let feed_forward_hidden_states = self.mlp.forward(&xs);
        attn_outputs + feed_forward_hidden_states + residual
    }

    fn gather_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn update_params(&self, _params: &mut HashMap<usize, Tensor>) {}

    fn gather_named_params(&self, _prefix: &str, _params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.self_attn
            .update_named_params(&prefix.push("self_attn"), params);
        self.mlp.update_named_params(&prefix.push("mlp"), params);
        self.input_layernorm
            .update_named_params(&prefix.push("input_layernorm"), params);
    }
}

trainable_module!(DecoderLayer);

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let embed_tokens = nn::Embedding::new(cfg.vocab_size, cfg.hidden_size, dtype, device);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| DecoderLayer::new(cfg, dtype, device))
            .collect();
        let final_layernorm =
            LayerNorm::new(cfg.hidden_size, cfg.layer_norm_eps, true, dtype, device);
        let lm_head = Linear::new(cfg.hidden_size, cfg.vocab_size, true, dtype, device);
        Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
        }
    }
}

impl Module for Model {
    type Input = Tensor; // x
    type Output = Tensor; // logits

    fn forward(&self, xs: &Self::Input) -> Self::Output {
        let [_b_size, seq_len]: [usize; 2] = xs.shape_of([0, 1]).try_into().unwrap();
        let mut xs = self.embed_tokens.forward(xs);
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(get_mask(seq_len, xs.device()))
        };
        for layer in &self.layers {
            xs = layer.forward(&(xs, mask.clone()));
        }
        let xs = self.final_layernorm.forward(&xs).narrow(1, seq_len - 1, 1);
        self.lm_head.forward(&xs).squeeze(1)
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

    fn gather_named_params(&self, _prefix: &str, _params: &mut HashMap<String, Tensor>) {
        todo!()
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        self.lm_head
            .update_named_params(&prefix.push("lm_head"), params);
        // split model and lm_head?
        let p = prefix.push("model");
        self.embed_tokens
            .update_named_params(&p.push("embed_tokens"), params);
        self.final_layernorm
            .update_named_params(&p.push("final_layernorm"), params);
        for (i, l) in self.layers.iter().enumerate() {
            let lp = format!("layers.{}", i);
            l.update_named_params(&p.push(&lp), params);
        }
    }
}

trainable_module!(Model);

fn load_model(dtype: impl Type, device: impl AsDevice) -> (Tokenizer, Model) {
    let start = Instant::now();
    let model_id = "microsoft/phi-2".to_string();
    let revision = "main".to_string();
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    // tokenizer
    let tokenizer_filename = repo.get("tokenizer.json").unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
    // model
    let config_filename = repo.get("config.json").unwrap();
    let config = std::fs::read_to_string(config_filename).unwrap();
    let cfg: Config = serde_json::from_str(&config).unwrap();
    let model_filenames = ext::hf::load_safetensors(&repo, "model.safetensors.index.json");
    let phi = Model::new(&cfg, dtype, device);
    phi.update_by_safetensors(&model_filenames);
    let elapsed = start.elapsed();
    println!("model loaded in : {:?}", elapsed);
    (tokenizer, phi)
}

fn sample_argmax(logits: &[f32]) -> u32 {
    let next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, u), (_, v)| u.total_cmp(v))
        .map(|(i, _)| i as u32)
        .unwrap();
    next_token
}

pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let context: std::collections::HashSet<_> = context.iter().collect();
    for (token_id, logit) in logits.iter_mut().enumerate() {
        if context.contains(&(token_id as u32)) {
            if *logit >= 0.0 {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
}

fn main() {
    let device: &dyn Device = if cuda_enabled() { &Cuda(0) } else { &Cpu };
    let dtype = F32;
    let (tokenizer, model) = load_model(dtype, device);

    let prompt = "A skier slides down a frictionless slope of height 40m and length 80m. What's the skier speed at the bottom?";
    println!("{prompt}");
    std::io::stdout().flush().unwrap();

    let tokens = tokenizer.encode(prompt, true).unwrap();
    let mut tokens = tokens.get_ids().to_vec();
    let mut generated_tokens = 0usize;
    let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
        Some(token) => *token,
        None => panic!("cannot find the endoftext token"),
    };

    let start_gen = std::time::Instant::now();
    let sample_len: usize = 5000;
    let repeat_penalty: f32 = 1.10;
    let repeat_last_n: usize = 64;

    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::from_array(ctxt, [ctxt.len()], device).unsqueeze(0);
        let logits = model.forward(&input);
        let logits = logits.squeeze(0);
        let mut logits = logits.as_vec(dtype);
        if repeat_penalty >= 1. {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            apply_repeat_penalty(&mut logits, repeat_penalty, &tokens[start_at..])
        };
        let next_token = sample_argmax(&logits);
        tokens.push(next_token);
        generated_tokens += 1;
        if next_token == eos_token {
            break;
        }
        let token = tokenizer.decode(&[next_token], true).unwrap();
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
}
