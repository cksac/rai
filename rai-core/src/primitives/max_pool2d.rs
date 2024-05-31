use crate::{Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct MaxPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl MaxPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }
}

impl Primitive for MaxPool2d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "MaxPool2d({:?}, {:?}, {:?}, {:?})",
            self.kernel_size, self.stride, self.padding, self.dilation
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for MaxPool2d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        assert_eq!(
            self.kernel_size, self.stride,
            "vjp not supported for maxpool2d if kernel_size != stride"
        );
        let x = &primals[0];
        let [_n, _c, h, w] = x.shape_before::<4>();
        let out_upsampled = &output.upsample_nearest2d([h, w]);
        let mask = x.eq(out_upsampled).to_dtype(x);
        let avg = mask.avg_pool2d((self.kernel_size, self.stride));
        let cotan_x = (cotangent * avg).upsample_nearest2d([h, w]) * mask;
        vec![cotan_x]
    }
}
