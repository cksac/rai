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
        let [_, _, cotan_h, cotan_w] = cotangent.shape_before::<4>();
        let [_, _, k_h, k_w] = kernel.shape_before::<4>();
        let out_h =
            (cotan_h - 1) * self.stride[0] + self.dilation[0] * (k_h - 1) + 1 - 2 * self.padding[0];
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
        let [_, _, ck_h, ck_w] = cotan_kernel.shape_before::<4>();
        let cotan_kernel = match (ck_h > k_h, ck_w > k_w) {
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
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let input = &primals[0];
        let kernel = &primals[1];
        let cotan_input = cotangent.conv2d(kernel, self.padding, self.stride, self.dilation, 1);
        let cotan_kernel = cotangent
            .transpose(0, 1)
            .conv2d(
                &input.transpose(0, 1),
                self.padding,
                self.stride,
                self.dilation,
                1,
            )
            .transpose(0, 1);
        let [_, _, k_h, k_w] = kernel.shape_before::<4>();
        let [_, _, ck_h, ck_w] = cotan_kernel.shape_before::<4>();
        let cotan_kernel = match (ck_h > k_h, ck_w > k_w) {
            (true, true) => cotan_kernel.narrow(2, 0, k_h).narrow(3, 0, k_w),
            (true, false) => cotan_kernel.narrow(2, 0, k_h),
            (false, true) => cotan_kernel.narrow(3, 0, k_w),
            (false, false) => cotan_kernel,
        };
        vec![cotan_input, cotan_kernel]
    }
}
