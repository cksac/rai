use crate::{dim::Before, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug, PartialEq)]
pub struct Conv2d {
    pub padding: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
}

impl Conv2d {
    pub fn new(padding: [usize; 2], stride: [usize; 2], dilation: [usize; 2]) -> Self {
        Self {
            padding,
            stride,
            dilation,
        }
    }
}

impl Op for Conv2d {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Conv2d")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "Conv2d({:?}, {:?}, {:?})",
            self.padding, self.stride, self.dilation
        )
    }

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for Conv2d")
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let [_, _, cotan_h, cotan_w] = cotangent.sizes(Before::<4>);
        let [_, _, k_h, k_w] = kernel.sizes(Before::<4>);
        let out_h =
            (cotan_h - 1) * self.stride[0] + self.dilation[0] * (k_h - 1) + 1 - 2 * self.padding[0];
        let out_w =
            (cotan_w - 1) * self.stride[1] + self.dilation[1] * (k_w - 1) + 1 - 2 * self.padding[1];
        let out_padding_h = input.size(2) - out_h;
        let out_padding_w = input.size(3) - out_w;
        let cotan_input = cotangent.conv_transpose2d(
            kernel,
            self.padding,
            [out_padding_h, out_padding_w],
            self.stride,
            self.dilation,
            1,
        );
        let cotan_kernel = input
            .transpose(0, 1)
            .conv2d(
                cotangent.transpose(0, 1),
                self.padding,
                self.stride,
                self.dilation,
                1,
            )
            .transpose(0, 1);
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
fn conv2d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv2d(kernel, &padding, &stride, &dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        Conv2d::new(padding, stride, dilation),
        inputs,
    )
}

#[track_caller]
pub fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    padding: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Tensor {
    let c_in = input.size(1);
    let c_in_k = kernel.size(1);
    assert_eq!(c_in, c_in_k * groups);
    if groups == 1 {
        conv2d_single_group(input, kernel, padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| conv2d_single_group(block, kernel, padding, stride, dilation))
            .collect::<Vec<_>>();
        Tensor::cat(&outputs, 1)
    }
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn conv2d(
        &self,
        kernel: impl AsRef<Tensor>,
        padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Tensor {
        conv2d(self, kernel.as_ref(), padding, stride, dilation, groups)
    }
}
