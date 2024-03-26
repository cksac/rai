use crate::{Primitive, Tensor};
use std::any::Any;
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

impl Primitive for FlashAttention {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
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
