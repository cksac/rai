use crate::{dim::Before, Op, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct ConvTranspose2d {
    pub padding: [usize; 2],
    pub out_padding: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
}

impl ConvTranspose2d {
    pub fn new(
        padding: [usize; 2],
        out_padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
    ) -> Self {
        Self {
            padding,
            out_padding,
            stride,
            dilation,
        }
    }
}

impl Op for ConvTranspose2d {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "ConvTranspose2d({:?}, {:?}, {:?}, {:?})",
            self.padding, self.out_padding, self.stride, self.dilation
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for ConvTranspose2d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let cotan_input = cotangent.conv2d(kernel, self.padding, self.stride, self.dilation, 1);
        let cotan_kernel = cotangent
            .transpose(0, 1)
            .conv2d(
                input.transpose(0, 1),
                self.padding,
                self.stride,
                self.dilation,
                1,
            )
            .transpose(0, 1);
        let [_, _, k_h, k_w] = kernel.sizes(Before::<4>);
        let [_, _, ck_h, ck_w] = cotan_kernel.sizes(Before::<4>);
        let cotan_kernel = match (ck_h > k_h, ck_w > k_w) {
            (true, true) => cotan_kernel.narrow(2, 0, k_h).narrow(3, 0, k_w),
            (true, false) => cotan_kernel.narrow(2, 0, k_h),
            (false, true) => cotan_kernel.narrow(3, 0, k_w),
            (false, false) => cotan_kernel,
        };
        vec![cotan_input, cotan_kernel]
    }
}

#[track_caller]
fn conv_transpose2d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    output_padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv_transpose2d(kernel, &padding, &output_padding, &stride, &dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        ConvTranspose2d::new(padding, output_padding, stride, dilation),
        inputs,
    )
}

#[track_caller]
pub fn conv_transpose2d(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    output_padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Tensor {
    if groups == 1 {
        conv_transpose2d_single_group(input, kernel, padding, output_padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| {
                conv_transpose2d_single_group(
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
