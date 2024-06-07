use crate::{Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::{any::Any, fmt::Debug};
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct FlashAttention {
    pub softmax_scale: f32,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub alibi_slopes: Option<Tensor>,
}

impl FlashAttention {
    pub fn new(
        softmax_scale: f32,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        alibi_slopes: Option<Tensor>,
    ) -> Self {
        Self {
            softmax_scale,
            window_size_left,
            window_size_right,
            alibi_slopes,
        }
    }

    pub fn softmax_scale(&self) -> f32 {
        self.softmax_scale
    }

    pub fn window_size_left(&self) -> Option<usize> {
        self.window_size_left
    }

    pub fn window_size_right(&self) -> Option<usize> {
        self.window_size_right
    }

    pub fn alibi_slopes(&self) -> Option<&Tensor> {
        self.alibi_slopes.as_ref()
    }
}

impl Op for FlashAttention {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "FlashAttention({}, {:?}, {:?}, {:?})",
            self.softmax_scale,
            self.window_size_left,
            self.window_size_right,
            self.alibi_slopes
                .as_ref()
                .map(|t| format!("tensor({})", t.id()))
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for FlashAttention")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!("vjp for FlashAttention")
    }
}

pub trait FlashAttentionOpts: Debug {
    fn softmax_scale(&self) -> f32;
    fn window_size_left(&self) -> Option<usize> {
        None
    }
    fn window_size_right(&self) -> Option<usize> {
        None
    }
    fn alibi_slopes(&self) -> Option<&Tensor> {
        None
    }
}

impl FlashAttentionOpts for f32 {
    fn softmax_scale(&self) -> f32 {
        *self
    }
}

impl<'a> FlashAttentionOpts for (f32, &'a Tensor) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn alibi_slopes(&self) -> Option<&Tensor> {
        Some(self.1)
    }
}

impl<'a> FlashAttentionOpts for (f32, usize, &'a Tensor) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn window_size_right(&self) -> Option<usize> {
        Some(self.1)
    }

    fn alibi_slopes(&self) -> Option<&Tensor> {
        Some(self.2)
    }
}

impl FlashAttentionOpts for (f32, usize, usize) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn window_size_right(&self) -> Option<usize> {
        Some(self.1)
    }

    fn window_size_left(&self) -> Option<usize> {
        Some(self.2)
    }
}

impl<'a> FlashAttentionOpts for (f32, usize, usize, &'a Tensor) {
    fn softmax_scale(&self) -> f32 {
        self.0
    }

    fn window_size_right(&self) -> Option<usize> {
        Some(self.1)
    }

    fn window_size_left(&self) -> Option<usize> {
        Some(self.2)
    }

    fn alibi_slopes(&self) -> Option<&Tensor> {
        Some(self.3)
    }
}

#[track_caller]
pub fn flash_attention(
    q: impl TryAsTensor,
    k: impl TryAsTensor,
    v: impl TryAsTensor,
    opts: impl FlashAttentionOpts,
) -> RaiResult<Tensor> {
    let q = crate::try_get! { q.try_as_tensor() };
    let k = crate::try_get! { k.try_as_tensor() };
    let v = crate::try_get! { v.try_as_tensor() };
    let device = q.device();
    let dtype = q.dtype();
    let shape = q.shape();
    let inputs = vec![q.clone(), k.clone(), v.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        FlashAttention::new(
            opts.softmax_scale(),
            opts.window_size_left(),
            opts.window_size_right(),
            opts.alibi_slopes().cloned(),
        ),
        inputs,
    )
    .into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn flash_attention(
        &self,
        k: impl TryAsTensor,
        v: impl TryAsTensor,
        opts: impl FlashAttentionOpts,
    ) -> RaiResult<Tensor> {
        flash_attention(self, k, v, opts)
    }
}
