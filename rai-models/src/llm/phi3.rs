use rai::{
    dim::Before,
    nn::{self, Activation, Embedding, Linear, Module, RmsNorm},
    AsDevice, Module, Shape, Tensor, Type, F32,
};
use serde::Deserialize;
use std::{cell::RefCell, fmt::Debug};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<String>,
    pub max_position_embeddings: usize,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, usize), trainable = false)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f64 / cfg.rope_theta.powf(i as f64 / dim as f64))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_array(inv_freq, [1, inv_freq_len], device).to_dtype(dtype);
        let t = Tensor::arange((0u32, max_seq_len as u32), device)
            .to_dtype(dtype)
            .reshape([max_seq_len, 1]);
        let freqs = t.matmul(&inv_freq);
        let emb = Tensor::cat(&[&freqs, &freqs], -1);
        Self {
            sin: emb.sin(),
            cos: emb.cos(),
            dim,
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
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
    #[param(skip)]
    i_size: usize,
}

impl MLP {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let gate_up_proj = Linear::new(hidden_size, 2 * i_size, false, dtype, device);
        let down_proj = Linear::new(i_size, hidden_size, false, dtype, device);
        Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
            i_size,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let up_states = x.apply(&self.gate_up_proj);
        let gate = up_states.narrow(-1, 0, self.i_size);
        let up_states = up_states.narrow(-1, self.i_size, self.i_size);
        let up_states = up_states * gate.apply(&self.act_fn);
        up_states.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Option<Tensor>, usize))]
pub struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    #[param(skip)]
    rotary_emb: RotaryEmbedding,
    #[param(skip)]
    num_heads: usize,
    #[param(skip)]
    num_kv_heads: usize,
    #[param(skip)]
    num_kv_groups: usize,
    #[param(skip)]
    head_dim: usize,
    #[param(skip)]
    kv_cache: RefCell<Option<(Tensor, Tensor)>>,
}

impl Attention {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        let qkv_proj = Linear::new(cfg.hidden_size, op_size, false, dtype, device);
        let o_proj = Linear::new(num_heads * head_dim, cfg.hidden_size, false, dtype, device);
        let rotary_emb = RotaryEmbedding::new(cfg, dtype, device);
        Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            kv_cache: RefCell::new(None),
        }
    }

    fn repeat_kv(&self, xs: Tensor, n_rep: usize) -> Tensor {
        if n_rep == 1 {
            xs
        } else {
            let [b_sz, num_kv_heads, seq_len, head_dim] = xs.sizes(Before::<4>);
            xs.unsqueeze(2)
                .broadcast_to([b_sz, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([b_sz, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }

    pub fn fwd(&self, xs: &Tensor, mask: Option<&Tensor>, seqlen_offset: usize) -> Tensor {
        let [b_size, seq_len, _n_embd] = xs.sizes(Before::<3>);
        let qkv = self.qkv_proj.fwd(xs);
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(-1, 0, query_pos);
        let key_states = qkv.narrow(-1, query_pos, self.num_kv_heads * self.head_dim);
        let value_states = qkv.narrow(
            -1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        );
        let query_states = query_states
            .reshape([b_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let key_states = key_states
            .reshape([b_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let value_states = value_states
            .reshape([b_size, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);

        let query_states = self.rotary_emb.forward(&(query_states, seqlen_offset));
        let key_states = self.rotary_emb.forward(&(key_states, seqlen_offset));
        let kv_cache = self.kv_cache.borrow();
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
        let key_states = self.repeat_kv(key_states, self.num_kv_groups);
        let value_states = self.repeat_kv(value_states, self.num_kv_groups);
        let attn_output = {
            let scale = 1f32 / f32::sqrt(self.head_dim as f32);
            let attn_weights = query_states.matmul(key_states.transpose(2, 3)) * scale;
            let attn_weights = match mask {
                None => attn_weights,
                Some(mask) => attn_weights + mask,
            };
            attn_weights.softmax(-1).matmul(value_states)
        };
        attn_output.transpose(1, 2).flatten(2..).apply(&self.o_proj)
    }

    pub fn clear_kv_cache(&self) {
        self.kv_cache.replace(None);
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Option<Tensor>, usize))]
pub struct DecoderLayer {
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

    pub fn fwd(&self, xs: &Tensor, mask: Option<&Tensor>, seqlen_offset: usize) -> Tensor {
        let residual = xs;
        let xs = self.input_layernorm.fwd(xs);
        let xs = self.self_attn.fwd(&xs, mask, seqlen_offset);
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
#[module(input = (Tensor, usize))]
pub struct Model {
    #[param(rename = "model.embed_tokens")]
    embed_tokens: Embedding,
    #[param(rename = "model.layers")]
    layers: Vec<DecoderLayer>,
    #[param(rename = "model.norm")]
    norm: RmsNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let embed_tokens = nn::Embedding::new(cfg.vocab_size, cfg.hidden_size, dtype, device);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| DecoderLayer::new(cfg, dtype, device))
            .collect();
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, dtype, device);
        let lm_head = Linear::new(cfg.hidden_size, cfg.vocab_size, false, dtype, device);
        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    fn prepare_decoder_attn_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Tensor {
        let device = self.lm_head.weight().device();
        let dtype = self.lm_head.weight().dtype();
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

    pub fn fwd(&self, xs: &Tensor, seqlen_offset: usize) -> Tensor {
        let [b_size, seq_len] = xs.sizes(Before::<2>);
        let mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attn_mask(b_size, seq_len, seqlen_offset);
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(xs);
        for layer in &self.layers {
            xs = layer.forward(&(xs, mask.clone(), seqlen_offset));
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
