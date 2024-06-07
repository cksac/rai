use crate::{dim::Before, try_get, Op, RaiResult, Shape, Tensor, TryAsTensor};
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

impl Op for ConvTranspose1d {
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
    input: impl TryAsTensor,
    kernel: impl TryAsTensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> RaiResult<Tensor> {
    let input = crate::try_get! { input.try_as_tensor() };
    let kernel = crate::try_get! { kernel.try_as_tensor() };
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
    .into()
}

#[track_caller]
pub fn conv_transpose1d(
    input: impl TryAsTensor,
    kernel: impl TryAsTensor,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> RaiResult<Tensor> {
    let input = crate::try_get! { input.try_as_tensor() };
    let kernel = crate::try_get! { kernel.try_as_tensor() };
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
                .to_std_result()
            })
            .collect::<Result<Vec<_>, _>>();
        let outputs = crate::try_get! { outputs };
        Tensor::cat(&outputs, 1).into()
    }
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn conv_transpose1d(
        &self,
        kernel: impl TryAsTensor,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> RaiResult<Tensor> {
        conv_transpose1d(
            self,
            kernel,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        )
    }
}
