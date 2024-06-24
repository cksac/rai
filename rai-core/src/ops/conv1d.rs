use crate::{Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

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

impl Op for Conv1d {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Conv1d")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
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

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for Conv1d")
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let l_in = cotangent.size(2);
        let k_size = kernel.size(2);
        let out_size =
            (l_in - 1) * self.stride + self.dilation * (k_size - 1) + 1 - 2 * self.padding;
        let out_padding = input.size(2) - out_size;
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
        let g_k_size = cotan_kernel.size(2);
        let cotan_kernel = if g_k_size > k_size {
            cotan_kernel.narrow(2, 0, k_size)
        } else {
            cotan_kernel
        };
        vec![cotan_input, cotan_kernel]
    }
}

#[track_caller]
fn conv1d_single_group(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Tensor {
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_conv1d(kernel, padding, stride, dilation);
    let inputs = vec![input.clone(), kernel.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        Conv1d::new(padding, stride, dilation),
        inputs,
    )
}

#[track_caller]
pub fn conv1d(
    input: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let c_in = input.size(1);
    let c_in_k = kernel.size(1);
    assert_eq!(c_in, c_in_k * groups);
    if groups == 1 {
        conv1d_single_group(input, kernel, padding, stride, dilation)
    } else {
        let blocks = input.chunk(groups, 1);
        let kernels = kernel.chunk(groups, 0);
        let outputs = blocks
            .iter()
            .zip(kernels.iter())
            .map(|(block, kernel)| conv1d_single_group(block, kernel, padding, stride, dilation))
            .collect::<Vec<_>>();
        Tensor::cat(&outputs, 1)
    }
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn conv1d(
        &self,
        kernel: impl AsRef<Tensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Tensor {
        conv1d(self, kernel.as_ref(), padding, stride, dilation, groups)
    }
}
