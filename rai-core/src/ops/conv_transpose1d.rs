use crate::{dim::Before, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

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

impl Op for ConvTranspose1d {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("ConvTranspose1d")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
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

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for ConvTranspose1d")
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let cotan_input = cotangent.conv1d(kernel, self.padding, self.stride, self.dilation, 1);
        let cotan_kernel = cotangent
            .transpose(0, 1)
            .conv1d(
                input.transpose(0, 1),
                self.padding,
                self.stride,
                self.dilation,
                1,
            )
            .transpose(0, 1);
        let [_, _, k_l] = kernel.sizes(Before::<3>);
        let [_, _, ck_l] = cotan_kernel.sizes(Before::<3>);
        let cotan_kernel = if ck_l > k_l {
            cotan_kernel.narrow(2, 0, k_l)
        } else {
            cotan_kernel
        };
        vec![cotan_input, cotan_kernel]
    }
}

#[track_caller]
fn conv_transpose1d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv_transpose1d(kernel, padding, output_padding, stride, dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ConvTranspose1d::new(padding, output_padding, stride, dilation),
        inputs,
    )
}

#[track_caller]
pub fn conv_transpose1d(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    if groups == 1 {
        conv_transpose1d_single_group(input, kernel, padding, output_padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| {
                conv_transpose1d_single_group(
                    block,
                    kernel,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                )
            })
            .collect::<Vec<_>>();
        Tensor::cat(&outputs, 1)
    }
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn conv_transpose1d(
        &self,
        kernel: impl AsRef<Tensor>,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Tensor {
        conv_transpose1d(
            self,
            kernel.as_ref(),
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        )
    }
}
