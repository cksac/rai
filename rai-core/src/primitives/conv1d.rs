use crate::{Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct Conv1d {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl Conv1d {
    pub fn new(padding: usize, stride: usize, dilation: usize) -> Self {
        Self {
            padding,
            stride,
            dilation,
        }
    }
}

impl Primitive for Conv1d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "Conv1d({}, {}, {})",
            self.padding, self.stride, self.dilation
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for Conv1d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let l_in = cotangent.shape_at(2);
        let k_size = kernel.shape_at(2);
        let out_size =
            (l_in - 1) * self.stride + self.dilation * (k_size - 1) + 1 - 2 * self.padding;
        let out_padding = input.shape_at(2) - out_size;
        let cotan_input = cotangent.conv_transpose1d(
            kernel,
            self.padding,
            out_padding,
            self.stride,
            self.dilation,
            1,
        );
        let cotan_kernel = input
            .transpose(0, 1)
            .conv1d(
                cotangent.transpose(0, 1),
                self.padding,
                self.stride,
                self.dilation,
                1,
            )
            .transpose(0, 1);
        let g_k_size = cotan_kernel.shape_at(2);
        let cotan_kernel = if g_k_size > k_size {
            cotan_kernel.narrow(2, 0, k_size)
        } else {
            cotan_kernel
        };
        vec![cotan_input, cotan_kernel]
    }
}
