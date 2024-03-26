use crate::{Primitive, Shape, Tensor};
use std::any::Any;
use tracing::Level;

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
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        assert_eq!(
            self.kernel_size, self.stride,
            "vjp not supported for maxpool1d if kernel_size != stride"
        );
        let x = &primals[0];
        let [_n, _c, l] = x.shape_before::<3>();
        let out_upsampled = &output.upsample_nearest1d(l);
        let mask = x.eq(out_upsampled).to_dtype(x);
        let avg = mask.avg_pool1d((self.kernel_size, self.stride));
        let cotan_x = (cotangent * avg).upsample_nearest1d(l) * mask;
        vec![cotan_x]
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

#[derive(Clone, Debug, PartialEq)]
pub struct AvgPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl AvgPool1d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Primitive for AvgPool1d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "AvgPool1d({}, {}, {})",
            self.kernel_size, self.stride, self.padding
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for AvgPool1d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        assert_eq!(
            self.kernel_size, self.stride,
            "vjp not supported for avgPool1d if kernel_size != stride"
        );
        let x = &primals[0];
        let [_n, _c, l] = x.shape_before::<3>();
        let cotan_upsampled = cotangent.upsample_nearest1d(l);
        let cotan_x = cotan_upsampled * (1.0f64 / self.kernel_size as f64);
        vec![cotan_x]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AvgPool2d {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl Primitive for AvgPool2d {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!(
            "AvgPool2d({:?}, {:?}, {:?})",
            self.kernel_size, self.stride, self.padding
        )
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for AvgPool2d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        assert_eq!(
            self.kernel_size, self.stride,
            "vjp not supported for avgPool2d if kernel_size != stride"
        );
        let x = &primals[0];
        let [_n, _c, h, w] = x.shape_before::<4>();
        let cotan_upsampled = cotangent.upsample_nearest2d([h, w]);
        let cotan_x = cotan_upsampled * (1.0f64 / (self.kernel_size.0 * self.kernel_size.1) as f64);
        vec![cotan_x]
    }
}
