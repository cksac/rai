use rai::{
    dim::Before,
    nn::{Activation, Embedding, Linear, Module},
    AsDType, AsDevice, Module, Shape, Tensor, Type, BF16, F16, F32,
};
use std::cell::RefCell;

fn default_max_position_embeddings() -> usize {
    4096
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone, Module)]
struct RmsNorm {
    weight: Tensor,
    #[param(skip)]
    eps: f64,
}

impl RmsNorm {
    pub fn new(dims: usize, eps: f64, dtype: impl Type, device: impl AsDevice) -> Self {
        let weight = Tensor::ones([dims], dtype, device);
        Self { weight, eps }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let x_dtype = x.dtype();
        let internal_dtype = if x_dtype == &F16 || x_dtype == &BF16 {
            &F32
        } else {
            x_dtype
        };
        let hidden_size = x.size(-1);
        let x = x.to_dtype(internal_dtype);
        let norm_x = x.square().sum((-1, true)) / hidden_size as f64;
        let x_normed = x / ((norm_x + self.eps).sqrt());
        x_normed.to_dtype(x_dtype) * (&self.weight + 1.0)
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Tensor, usize), output = (Tensor, Tensor), trainable = false)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

fn rotate_half(xs: &Tensor) -> Tensor {
    let last_dim = xs.size(-1);
    let xs1 = xs.narrow(-1, 0, last_dim / 2);
    let xs2 = xs.narrow(-1, last_dim / 2, last_dim - last_dim / 2);
    Tensor::cat(&[&xs2.neg(), &xs1], -1)
}

impl RotaryEmbedding {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_array(inv_freq, [1, inv_freq_len], device).to_dtype(dtype);
        let t = Tensor::arange((0u32, max_seq_len as u32), device)
            .to_dtype(dtype)
            .reshape([max_seq_len, 1]);
        let freqs = t.matmul(&inv_freq);
        let freqs = Tensor::cat(&[&freqs, &freqs], -1);
        Self {
            sin: freqs.sin(),
            cos: freqs.cos(),
        }
    }

    pub fn fwd(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> (Tensor, Tensor) {
        let [_b_sz, _h, seq_len, _n_embd] = q.sizes(Before::<4>);
        let cos = self.cos.narrow(0, seqlen_offset, seq_len);
        let sin = self.sin.narrow(0, seqlen_offset, seq_len);
        let cos = &cos.unsqueeze(0).unsqueeze(0); // (1, 1, seq_len, dim)
        let sin = &sin.unsqueeze(0).unsqueeze(0); // (1, 1, seq_len, dim)
        let q_embed = q * cos + rotate_half(q) * sin;
        let k_embed = k * cos + rotate_half(k) * sin;
        (q_embed, k_embed)
    }
}

#[derive(Debug, Clone, Module)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let gate_proj = Linear::new(hidden_size, intermediate_size, false, dtype, device);
        let up_proj = Linear::new(hidden_size, intermediate_size, false, dtype, device);
        let down_proj = Linear::new(intermediate_size, hidden_size, false, dtype, device);
        let act_fn = cfg.hidden_act;
        Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        }
    }

    pub fn fwd(&self, xs: &Tensor) -> Tensor {
        let lhs = xs.apply(&self.gate_proj).apply(&self.act_fn);
        let rhs = xs.apply(&self.up_proj);
        (lhs * rhs).apply(&self.down_proj)
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Option<Tensor>, usize))]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    #[param(skip)]
    num_heads: usize,
    #[param(skip)]
    num_kv_heads: usize,
    #[param(skip)]
    num_kv_groups: usize,
    #[param(skip)]
    head_dim: usize,
    #[param(skip)]
    hidden_size: usize,
    #[param(skip)]
    rotary_emb: RotaryEmbedding,
    #[param(skip)]
    kv_cache: RefCell<Option<(Tensor, Tensor)>>,
}

impl Attention {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_size / num_heads;
        let bias = cfg.attention_bias;
        let q_proj = Linear::new(hidden_size, num_heads * head_dim, bias, dtype, device);
        let k_proj = Linear::new(hidden_size, num_kv_heads * head_dim, bias, dtype, device);
        let v_proj = Linear::new(hidden_size, num_kv_heads * head_dim, bias, dtype, device);
        let o_proj = Linear::new(num_heads * head_dim, hidden_size, bias, dtype, device);
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, device);
        let kv_cache = RefCell::new(None);
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
        }
    }

    fn repeat_kv(&self, xs: Tensor) -> Tensor {
        let n_rep = self.num_kv_groups;
        if n_rep == 1 {
            xs
        } else {
            let [b_sz, num_kv_heads, seq_len, head_dim] = xs.sizes(Before::<4>);
            xs.unsqueeze(2)
                .broadcast_to([b_sz, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([b_sz, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }

    pub fn fwd(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Tensor {
        let [b_sz, q_len] = xs.sizes(Before::<2>);
        let query_states = self.q_proj.forward(xs);
        let key_states = self.k_proj.forward(xs);
        let value_states = self.v_proj.forward(xs);
        let query_states = query_states
            .reshape([b_sz, q_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let key_states = key_states
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let value_states = value_states
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let (query_states, key_states) =
            self.rotary_emb
                .fwd(&query_states, &key_states, seqlen_offset);

        let kv_cache = self.kv_cache.borrow();
        let (key_states, value_states) = match &*kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2);
                let value_states = Tensor::cat(&[prev_v, &value_states], 2);
                (key_states, value_states)
            }
        };
        drop(kv_cache);
        self.kv_cache
            .replace(Some((key_states.clone(), value_states.clone())));
        let key_states = self.repeat_kv(key_states);
        let value_states = self.repeat_kv(value_states);
        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = query_states.matmul(key_states.transpose(2, 3)) * scale;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights + mask,
            };
            let attn_weights = attn_weights.softmax(-1);
            attn_weights.matmul(&value_states)
        };
        attn_output
            .transpose(1, 2)
            .reshape([b_sz, q_len, self.hidden_size])
            .apply(&self.o_proj)
    }

    pub fn clear_kv_cache(&self) {
        self.kv_cache.replace(None);
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Option<Tensor>, usize))]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let self_attn = Attention::new(cfg, dtype, device);
        let mlp = MLP::new(cfg, dtype, device);
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, dtype, device);
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, dtype, device);
        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    pub fn fwd(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Tensor {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs);
        let xs = self.self_attn.fwd(&xs, attention_mask, seqlen_offset);
        let xs = xs + residual;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm).apply(&self.mlp);
        residual + xs
    }

    pub fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor,usize))]
pub struct Model {
    #[param(rename = "model.embed_tokens")]
    embed_tokens: Embedding,
    #[param(rename = "model.layers")]
    layers: Vec<DecoderLayer>,
    #[param(rename = "model.norm")]
    norm: RmsNorm,
    // share weight with embed_tokens's weight
    #[param(skip)]
    lm_head: Linear,
    #[param(skip)]
    hidden_size: usize,
}

impl Model {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let embed_tokens = Embedding::new(cfg.vocab_size, cfg.hidden_size, dtype, device);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| DecoderLayer::new(cfg, dtype, device))
            .collect();
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, dtype, device);
        let lm_head = Linear::new_with_params(embed_tokens.weight().clone(), None);
        let hidden_size = cfg.hidden_size;
        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size,
        }
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
        dtype: impl AsDType,
        device: impl AsDevice,
    ) -> Tensor {
        let device = device.device();
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_array(mask, [tgt_len, tgt_len], device);
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros([tgt_len, seqlen_offset], F32, device);
            Tensor::cat(&[&mask0, &mask], -1)
        } else {
            mask
        };
        mask.broadcast_to([b_size, 1, tgt_len, tgt_len + seqlen_offset])
            .to_dtype(dtype)
    }

    pub fn fwd(&self, input: &Tensor, seqlen_offset: usize) -> Tensor {
        let [b_size, seq_len] = input.sizes(Before::<2>);
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(
                b_size,
                seq_len,
                seqlen_offset,
                self.embed_tokens.weight().dtype(),
                self.embed_tokens.weight().device(),
            );
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input);
        xs = xs * (self.hidden_size as f64).sqrt();
        for layer in &self.layers {
            xs = layer.fwd(&xs, attention_mask.as_ref(), seqlen_offset);
        }
        xs.narrow(1, seq_len - 1, 1)
            .apply(&self.norm)
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&self) {
        for layer in self.layers.iter() {
            layer.clear_kv_cache()
        }
    }
}
