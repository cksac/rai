use crate::{Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct UpsampleNearest1d {
    pub size: usize,
}

impl UpsampleNearest1d {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Primitive for UpsampleNearest1d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("UpsampleNearest1d({})", self.size)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for UpsampleNearest1d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let [_n, c, size] = x.shape_before::<3>();
        assert!(
            self.size % size != 0,
            "UpsampleNearest1d vjp not supported for non integer upscaling factors"
        );
        let scale = self.size / size;
        let kernel = &Tensor::ones([c, 1, scale], x.dtype(), x.device());
        let cotan_x = cotangent.conv1d(kernel, 0, scale, 1, c);
        vec![cotan_x]
    }
}
