use rai::{
    dim::Before,
    nn::{self, Activation, Embedding, LayerNorm, Linear, Module},
    AsDevice, Module, Shape, Tensor, Type, F32,
};
use serde::Deserialize;
use std::{cell::RefCell, fmt::Debug};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: Activation,
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

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, usize), trainable = false)]
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

    pub fn fwd(&self, xs: &Tensor, seqlen_offset: usize) -> Tensor {
        let [_b_size, _num_heads, seq_len, headdim] = xs.sizes(Before::<4>);
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
}

#[derive(Debug, Clone, Module)]
pub struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl MLP {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let fc1 = Linear::new(cfg.hidden_size, cfg.intermediate_size, true, dtype, device);
        let fc2 = Linear::new(cfg.intermediate_size, cfg.hidden_size, true, dtype, device);
        Self {
            fc1,
            fc2,
            act: cfg.hidden_act,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        x.apply(&self.fc1).apply(&self.act).apply(&self.fc2)
    }
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

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Option<Tensor>))]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    #[param(skip)]
    rotary_emb: RotaryEmbedding,
    #[param(skip)]
    softmax_scale: f64,
    #[param(skip)]
    num_heads: usize,
    #[param(skip)]
    num_kv_heads: usize,
    #[param(skip)]
    head_dim: usize,
    #[param(skip)]
    kv_cache: RefCell<Option<(Tensor, Tensor)>>,
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
            let [b_sz, num_kv_heads, seq_len, head_dim] = xs.sizes(Before::<4>);
            xs.unsqueeze(2)
                .broadcast_to([b_sz, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([b_sz, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }

    pub fn fwd(&self, xs: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let [b_size, seq_len, _n_embd] = xs.sizes(Before::<3>);
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
            Some((prev_k, _)) => prev_k.size(2),
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
        let key_states = self.repeat_kv(key_states);
        let value_states = self.repeat_kv(value_states);

        // Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        let attn_weights = query_states
            .to_dtype(F32)
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
        let attn_output = attn_output.reshape([b_size, seq_len, attn_output.dims_elem_count(2..)]);
        self.dense.forward(&attn_output)
    }

    pub fn clear_kv_cache(&self) {
        self.kv_cache.replace(None);
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Option<Tensor>))]
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

    pub fn fwd(&self, xs: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs);
        let attn_outputs = self.self_attn.forward(&(xs.clone(), mask.cloned()));
        let feed_forward_hidden_states = self.mlp.forward(&xs);
        attn_outputs + feed_forward_hidden_states + residual
    }

    pub fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache()
    }
}
#[derive(Debug, Clone, Module)]
pub struct Model {
    #[param(rename = "model.embed_tokens")]
    embed_tokens: Embedding,
    #[param(rename = "model.layers")]
    layers: Vec<DecoderLayer>,
    #[param(rename = "model.final_layernorm")]
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

    pub fn fwd(&self, xs: &Tensor) -> Tensor {
        let [_b_size, seq_len] = xs.sizes(Before::<2>);
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

    pub fn clear_kv_cache(&self) {
        for layer in self.layers.iter() {
            layer.clear_kv_cache()
        }
    }
}
