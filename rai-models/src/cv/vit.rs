use rai::{
    nn::{Activation, ApplyModule, Conv2d, Conv2dConfig, LayerNorm, Linear, Module},
    AsDevice, Module, Shape, Tensor, Type,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub layer_norm_eps: f64,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub qkv_bias: bool,
    #[serde(default)]
    pub use_mask_token: bool,
}

#[derive(Debug, Clone, Module)]
struct PatchEmbeddings {
    projection: Conv2d,
    #[param(skip)]
    num_patches: usize,
}

impl PatchEmbeddings {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let image_size = cfg.image_size;
        let patch_size = cfg.patch_size;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let conv_cfg = Conv2dConfig {
            stride: [patch_size, patch_size],
            ..Default::default()
        };
        let projection = Conv2d::new(
            cfg.num_channels,
            cfg.hidden_size,
            patch_size,
            conv_cfg,
            true,
            dtype,
            device,
        );
        Self {
            projection,
            num_patches,
        }
    }

    pub fn fwd(&self, pixel_values: &Tensor) -> Tensor {
        self.projection
            .forward(pixel_values)
            .flatten(2)
            .transpose(1, 2)
    }
}

#[derive(Debug, Clone, Module)]
#[module(input=(Tensor, Option<Tensor>, bool))]
pub struct Embeddings {
    cls_token: Tensor,
    mask_token: Option<Tensor>,
    patch_embeddings: PatchEmbeddings,
    position_embeddings: Tensor,
    #[param(skip)]
    hidden_size: usize,
}

impl Embeddings {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let hidden_size = cfg.hidden_size;
        let cls_token = Tensor::rand([1, 1, hidden_size], dtype, device);
        let mask_token = if cfg.use_mask_token {
            Some(Tensor::rand([1, 1, hidden_size], dtype, device))
        } else {
            None
        };
        let patch_embeddings = PatchEmbeddings::new(cfg, dtype, device);
        let num_patches = patch_embeddings.num_patches;
        let position_embeddings = Tensor::rand([1, num_patches + 1, hidden_size], dtype, device);
        Self {
            cls_token,
            mask_token,
            patch_embeddings,
            position_embeddings,
            hidden_size,
        }
    }

    fn interpolate_pos_encoding(
        &self,
        _embeddings: &Tensor,
        _height: usize,
        _width: usize,
    ) -> Tensor {
        // https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L82
        todo!("interpolate_pos_encoding")
    }

    pub fn fwd(
        &self,
        pixel_values: &Tensor,
        bool_masked_pos: Option<&Tensor>,
        interpolate_pos_encoding: bool,
    ) -> Tensor {
        let [b_size, _num_channels, height, width] = pixel_values.shape_before::<4>();
        let embeddings = self.patch_embeddings.forward(pixel_values);
        let embeddings = match (bool_masked_pos, &self.mask_token) {
            (None, _) => embeddings,
            (Some(_), None) => panic!("bool_masked_pos set without mask_token"),
            (Some(bool_masked_pos), Some(mask_tokens)) => {
                let seq_len = embeddings.shape_at(1);
                let mask_tokens = mask_tokens.broadcast_to([b_size, seq_len, self.hidden_size]);
                let mask = &bool_masked_pos.unsqueeze(-1).to_dtype(&mask_tokens);
                mask_tokens * mask - embeddings * (mask - 1.0)
            }
        };
        let cls_tokens = self.cls_token.broadcast_to([b_size, 1, self.hidden_size]);
        let embeddings = Tensor::cat(&[cls_tokens, embeddings], 1);
        if interpolate_pos_encoding {
            let pos = self.interpolate_pos_encoding(&embeddings, height, width);
            embeddings + pos
        } else {
            embeddings + &self.position_embeddings
        }
    }
}

#[derive(Debug, Clone, Module)]
pub struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    #[param(skip)]
    num_attention_heads: usize,
    #[param(skip)]
    attention_head_size: usize,
}

impl SelfAttention {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let num_attention_heads = cfg.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;
        let query = Linear::new(cfg.hidden_size, all_head_size, cfg.qkv_bias, dtype, device);
        let key = Linear::new(cfg.hidden_size, all_head_size, cfg.qkv_bias, dtype, device);
        let value = Linear::new(cfg.hidden_size, all_head_size, cfg.qkv_bias, dtype, device);
        Self {
            query,
            key,
            value,
            num_attention_heads,
            attention_head_size,
        }
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Tensor {
        let [b_size, seq_len] = xs.shape_before::<2>();
        xs.reshape([
            b_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ])
        .permute([0, 2, 1, 3])
    }

    pub fn fwd(&self, xs: &Tensor) -> Tensor {
        let query = self.query.forward(xs);
        let key = self.key.forward(xs);
        let value = self.value.forward(xs);
        let query = self.transpose_for_scores(&query).to_contiguous();
        let key = self.transpose_for_scores(&key).to_contiguous();
        let value = self.transpose_for_scores(&value).to_contiguous();
        let attention_scores = query.matmul(key.t()) / f64::sqrt(self.attention_head_size as f64);
        let attention_probs = attention_scores.softmax(-1);
        attention_probs
            .matmul(value)
            .permute([0, 2, 1, 3])
            .to_contiguous()
            .flatten(2)
    }
}

#[derive(Debug, Clone, Module)]
pub struct SelfOutput {
    dense: Linear,
}

impl SelfOutput {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let dense = Linear::new(cfg.hidden_size, cfg.hidden_size, true, dtype, device);
        Self { dense }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        self.dense.forward(x)
    }
}

#[derive(Debug, Clone, Module)]
struct Attention {
    attention: SelfAttention,
    output: SelfOutput,
}

impl Attention {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let attention = SelfAttention::new(cfg, dtype, device);
        let output = SelfOutput::new(cfg, dtype, device);
        Self { attention, output }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        x.apply(&self.attention).apply(&self.output)
    }
}

#[derive(Debug, Clone, Module)]
pub struct Intermediate {
    dense: Linear,
    intermediate_act_fn: Activation,
}

impl Intermediate {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let dense = Linear::new(cfg.hidden_size, cfg.intermediate_size, true, dtype, device);
        Self {
            dense,
            intermediate_act_fn: cfg.hidden_act,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        x.apply(&self.dense).apply(&self.intermediate_act_fn)
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Tensor))]
pub struct Output {
    dense: Linear,
}

impl Output {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let dense = Linear::new(cfg.intermediate_size, cfg.hidden_size, true, dtype, device);
        Self { dense }
    }

    pub fn fwd(&self, xs: &Tensor, input_tensor: &Tensor) -> Tensor {
        self.dense.forward(xs) + input_tensor
    }
}

#[derive(Debug, Clone, Module)]
struct Layer {
    attention: Attention,
    intermediate: Intermediate,
    output: Output,
    layernorm_before: LayerNorm,
    layernorm_after: LayerNorm,
}

impl Layer {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let attention = Attention::new(cfg, dtype, device);
        let intermediate = Intermediate::new(cfg, dtype, device);
        let output = Output::new(cfg, dtype, device);
        let h_sz = cfg.hidden_size;
        let ln_eps = cfg.layer_norm_eps;
        let layernorm_before = LayerNorm::new(h_sz, ln_eps, true, dtype, device);
        let layernorm_after = LayerNorm::new(h_sz, ln_eps, true, dtype, device);
        Self {
            attention,
            intermediate,
            output,
            layernorm_before,
            layernorm_after,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.layernorm_before).apply(&self.attention) + x;
        let y = x.apply(&self.layernorm_after).apply(&self.intermediate);
        self.output.forward(&(y, x))
    }
}

#[derive(Debug, Clone, Module)]
pub struct Encoder {
    layers: Vec<Layer>,
}

impl Encoder {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = Layer::new(cfg, dtype, device);
            layers.push(layer)
        }
        Self { layers }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for layer in &self.layers {
            x = x.apply(layer);
        }
        x
    }
}

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, Option<Tensor>, bool))]
pub struct Model {
    #[param(rename = "vit.embeddings")]
    embeddings: Embeddings,
    #[param(rename = "vit.encoder")]
    encoder: Encoder,
    #[param(rename = "vit.layernorm")]
    layernorm: LayerNorm,
}

impl Model {
    pub fn new(cfg: &Config, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let embeddings = Embeddings::new(cfg, dtype, device);
        let encoder = Encoder::new(cfg, dtype, device);
        let layernorm = LayerNorm::new(cfg.hidden_size, cfg.layer_norm_eps, true, dtype, device);
        Self {
            embeddings,
            encoder,
            layernorm,
        }
    }

    pub fn fwd(
        &self,
        x: &Tensor,
        bool_masked_pos: Option<&Tensor>,
        interpolate_pos_encoding: bool,
    ) -> Tensor {
        let embedding_output = self
            .embeddings
            .fwd(x, bool_masked_pos, interpolate_pos_encoding);
        let encoder_outputs = self.encoder.fwd(&embedding_output);
        let [s1, _s2, s3] = embedding_output.shape_before::<3>();
        embedding_output
            .narrow(0, 0, s1)
            .narrow(1, 0, 0)
            .narrow(2, 0, s3)
            .apply(&self.layernorm)
    }
}

#[derive(Debug, Clone, Module)]
pub struct ImageClassificationModel {
    vit: Model,
    classifier: Linear,
}

impl ImageClassificationModel {
    pub fn new(cfg: &Config, num_labels: usize, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let vit = Model::new(cfg, dtype, device);
        let classifier = Linear::new(cfg.hidden_size, num_labels, true, dtype, device);
        Self { vit, classifier }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        self.vit.fwd(x, None, false).apply(&self.classifier)
    }
}
