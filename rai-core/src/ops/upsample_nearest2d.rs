use super::ToPair;
use crate::{dim::Before, Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct UpsampleNearest2d {
    pub size: (usize, usize),
}

impl UpsampleNearest2d {
    pub fn new(size: (usize, usize)) -> Self {
        Self { size }
    }
}

impl Op for UpsampleNearest2d {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("UpsampleNearest2d({:?})", self.size)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for UpsampleNearest2d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let [_n, c, h, w] = x.sizes(Before::<4>);
        assert!(
            self.size.0 % h != 0 || self.size.1 % w != 0,
            "UpsampleNearest2d vjp not supported for non integer upscaling factors"
        );
        let scale_h = self.size.0 / h;
        let scale_w = self.size.1 / w;
        let kernel = Tensor::ones([c, 1, scale_h, scale_w], x.dtype(), x.device());
        let cotan_x = cotangent.conv2d(kernel, [0, 0], [scale_h, scale_w], [1, 1], c);
        vec![cotan_x]
    }
}

#[track_caller]
pub fn upsample_nearest2d(input: &Tensor, size: impl ToPair<usize>) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let size = size.to_pair();
    let shape = input.shape_upsample_nearest2d(&size);
    let inputs = vec![input.clone()];
    Tensor::new(device, dtype, shape, UpsampleNearest2d::new(size), inputs)
}
