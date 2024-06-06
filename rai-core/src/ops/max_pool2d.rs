use super::ToPair;
use crate::{dim::Before, Op, Shape, Tensor};
use std::{any::Any, fmt::Debug};
use tracing::Level;

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

impl Op for MaxPool2d {
    fn clone_boxed(&self) -> Box<dyn Op> {
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
        let [_n, _c, h, w] = x.sizes(Before::<4>);
        let out_upsampled = &output.upsample_nearest2d([h, w]);
        let mask = x.eq(out_upsampled).to_dtype(x);
        let avg = mask.avg_pool2d((self.kernel_size, self.stride));
        let cotan_x = (cotangent * avg).upsample_nearest2d([h, w]) * mask;
        vec![cotan_x]
    }
}

pub trait MaxPool2dArgs: Debug {
    fn kernel_size(&self) -> (usize, usize);
    fn stride(&self) -> (usize, usize) {
        self.kernel_size()
    }
    fn padding(&self) -> (usize, usize) {
        (0, 0)
    }
    fn dilation(&self) -> (usize, usize) {
        (1, 1)
    }
}

impl MaxPool2dArgs for usize {
    fn kernel_size(&self) -> (usize, usize) {
        self.to_pair()
    }
}

impl MaxPool2dArgs for [usize; 2] {
    fn kernel_size(&self) -> (usize, usize) {
        self.to_pair()
    }
}

impl<A> MaxPool2dArgs for (A,)
where
    A: ToPair<usize>,
{
    fn kernel_size(&self) -> (usize, usize) {
        self.0.to_pair()
    }
}

impl<A, B> MaxPool2dArgs for (A, B)
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

impl<A, B, C> MaxPool2dArgs for (A, B, C)
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

impl<A, B, C, D> MaxPool2dArgs for (A, B, C, D)
where
    A: ToPair<usize>,
    B: ToPair<usize>,
    C: ToPair<usize>,
    D: ToPair<usize>,
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

    fn dilation(&self) -> (usize, usize) {
        self.3.to_pair()
    }
}

#[track_caller]
pub fn max_pool2d(input: &Tensor, args: impl MaxPool2dArgs) -> Tensor {
    let kernel_size = args.kernel_size();
    let stride = args.stride();
    let padding = args.padding();
    let dilation = args.dilation();
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_max_pool2d(&kernel_size, &stride, &padding, &dilation);
    let inputs = vec![input.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        MaxPool2d::new(kernel_size, stride, padding, dilation),
        inputs,
    )
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn max_pool2d(&self, args: impl MaxPool2dArgs) -> Tensor {
        max_pool2d(self, args)
    }
}
