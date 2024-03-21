use crate::{Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct FlashAttention {
    pub softmax_scale: f32,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub alibi_slopes: Option<Tensor>,
}

impl FlashAttention {
    pub fn new(
        softmax_scale: f32,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        alibi_slopes: Option<Tensor>,
    ) -> Self {
        Self {
            softmax_scale,
            window_size_left,
            window_size_right,
            alibi_slopes,
        }
    }

    pub fn softmax_scale(&self) -> f32 {
        self.softmax_scale
    }

    pub fn window_size_left(&self) -> Option<usize> {
        self.window_size_left
    }

    pub fn window_size_right(&self) -> Option<usize> {
        self.window_size_right
    }

    pub fn alibi_slopes(&self) -> Option<&Tensor> {
        self.alibi_slopes.as_ref()
    }
}

impl Primitive for FlashAttention {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "FlashAttention({}, {:?}, {:?}, {:?})",
            self.softmax_scale,
            self.window_size_left,
            self.window_size_right,
            self.alibi_slopes
                .as_ref()
                .map(|t| format!("tensor({})", t.id()))
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for FlashAttention")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!("vjp for FlashAttention")
    }
}

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

impl Primitive for Conv2d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
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

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for Conv2d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let cotan_h = cotangent.shape_at(2);
        let k_h = kernel.shape_at(2);
        let out_h =
            (cotan_h - 1) * self.stride[0] + self.dilation[0] * (k_h - 1) + 1 - 2 * self.padding[0];
        let cotan_w = cotangent.shape_at(3);
        let k_w = kernel.shape_at(3);
        let out_w =
            (cotan_w - 1) * self.stride[1] + self.dilation[1] * (k_w - 1) + 1 - 2 * self.padding[1];
        let out_padding_h = input.shape_at(2) - out_h;
        let out_padding_w = input.shape_at(3) - out_w;
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

        let g_k_h = cotan_kernel.shape_at(2);
        let g_k_w = cotan_kernel.shape_at(3);
        let cotan_kernel = match (g_k_h > k_h, g_k_w > k_w) {
            (true, true) => cotan_kernel.narrow(2, 0, k_h).narrow(3, 0, k_w),
            (true, false) => cotan_kernel.narrow(2, 0, k_h),
            (false, true) => cotan_kernel.narrow(3, 0, k_w),
            (false, false) => cotan_kernel,
        };
        vec![cotan_input, cotan_kernel]
    }
}

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
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!("vjp for ConvTranspose1d")
    }
}

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

impl Primitive for ConvTranspose2d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
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
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!("vjp for ConvTranspose2d")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaxPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl MaxPool1d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }
}

impl Primitive for MaxPool1d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "MaxPool1d({}, {}, {}, {})",
            self.kernel_size, self.stride, self.padding, self.dilation
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for MaxPool1d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!("vjp for MaxPool1d")
    }
}

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
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        todo!("vjp for MaxPool2d")
    }
}
