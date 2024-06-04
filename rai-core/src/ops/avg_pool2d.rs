use super::ToPair;
use crate::dim::Before;
use crate::{Op, Shape, Tensor};
use std::any::Any;
use std::fmt::Debug;
use tracing::Level;

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

impl Op for AvgPool2d {
    fn clone_boxed(&self) -> Box<dyn Op> {
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
        let [_n, _c, h, w] = x.sizes(Before::<4>);
        let cotan_upsampled = cotangent.upsample_nearest2d([h, w]);
        let cotan_x = cotan_upsampled * (1.0f32 / (self.kernel_size.0 * self.kernel_size.1) as f32);
        vec![cotan_x]
    }
}

pub trait AvgPool2dArgs: Debug {
    fn kernel_size(&self) -> (usize, usize);
    fn stride(&self) -> (usize, usize) {
        self.kernel_size()
    }
    fn padding(&self) -> (usize, usize) {
        (0, 0)
    }
}

impl AvgPool2dArgs for usize {
    fn kernel_size(&self) -> (usize, usize) {
        self.to_pair()
    }
}

impl AvgPool2dArgs for [usize; 2] {
    fn kernel_size(&self) -> (usize, usize) {
        self.to_pair()
    }
}

impl<A> AvgPool2dArgs for (A,)
where
    A: ToPair<usize>,
{
    fn kernel_size(&self) -> (usize, usize) {
        self.0.to_pair()
    }
}

impl<A, B> AvgPool2dArgs for (A, B)
where
    A: ToPair<usize>,
    B: ToPair<usize>,
{
    fn kernel_size(&self) -> (usize, usize) {
        self.0.to_pair()
    }

    fn stride(&self) -> (usize, usize) {
        self.1.to_pair()
    }
}

impl<A, B, C> AvgPool2dArgs for (A, B, C)
where
    A: ToPair<usize>,
    B: ToPair<usize>,
    C: ToPair<usize>,
{
    fn kernel_size(&self) -> (usize, usize) {
        self.0.to_pair()
    }

    fn stride(&self) -> (usize, usize) {
        self.1.to_pair()
    }

    fn padding(&self) -> (usize, usize) {
        self.2.to_pair()
    }
}

#[track_caller]
pub fn avg_pool2d(input: &Tensor, args: impl AvgPool2dArgs) -> Tensor {
    let kernel_size = args.kernel_size();
    let stride = args.stride();
    let padding = args.padding();

    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_avg_pool2d(&kernel_size, &stride, &padding);
    let inputs = vec![input.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        AvgPool2d::new(kernel_size, stride, padding),
        inputs,
    )
}
