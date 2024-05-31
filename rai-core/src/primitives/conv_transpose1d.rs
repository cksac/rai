use crate::{Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct ConvTranspose1d {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl ConvTranspose1d {
    pub fn new(padding: usize, output_padding: usize, stride: usize, dilation: usize) -> Self {
        Self {
            padding,
            output_padding,
            stride,
            dilation,
        }
    }
}

impl Primitive for ConvTranspose1d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "ConvTranspose1d({}, {}, {}, {})",
            self.padding, self.output_padding, self.stride, self.dilation
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for ConvTranspose1d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let cotan_input = cotangent.conv1d(kernel, self.padding, self.stride, self.dilation, 1);
        let cotan_kernel = cotangent
            .transpose(0, 1)
            .conv1d(
                &input.transpose(0, 1),
                self.padding,
                self.stride,
                self.dilation,
                1,
            )
            .transpose(0, 1);
        let [_, _, k_l] = kernel.shape_before::<3>();
        let [_, _, ck_l] = cotan_kernel.shape_before::<3>();
        let cotan_kernel = if ck_l > k_l {
            cotan_kernel.narrow(2, 0, k_l)
        } else {
            cotan_kernel
        };
        vec![cotan_input, cotan_kernel]
    }
}
