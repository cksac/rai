use crate::{dim::Before, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::{any::Any, fmt::Debug};
use tracing::Level;

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

impl Op for AvgPool1d {
    fn clone_boxed(&self) -> Box<dyn Op> {
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
        let [_n, _c, l] = x.sizes(Before::<3>);
        let cotan_upsampled = cotangent.upsample_nearest1d(l);
        let cotan_x = cotan_upsampled * (1.0f32 / self.kernel_size as f32);
        vec![cotan_x]
    }
}

pub trait AvgPool1dArgs: Debug {
    fn kernel_size(&self) -> usize;
    fn stride(&self) -> usize {
        self.kernel_size()
    }
    fn padding(&self) -> usize {
        0
    }
}

impl AvgPool1dArgs for usize {
    fn kernel_size(&self) -> usize {
        *self
    }
}

impl AvgPool1dArgs for (usize, usize) {
    fn kernel_size(&self) -> usize {
        self.0
    }

    fn stride(&self) -> usize {
        self.1
    }
}

impl AvgPool1dArgs for (usize, usize, usize) {
    fn kernel_size(&self) -> usize {
        self.0
    }

    fn stride(&self) -> usize {
        self.1
    }

    fn padding(&self) -> usize {
        self.2
    }
}

#[track_caller]
pub fn avg_pool1d(input: impl TryAsTensor, args: impl AvgPool1dArgs) -> RaiResult<Tensor> {
    let input = crate::try_get! { input.try_as_tensor() };
    let kernel_size = args.kernel_size();
    let stride = args.stride();
    let padding = args.padding();
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_avg_pool1d(kernel_size, stride, padding);
    let inputs = vec![input.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        AvgPool1d::new(kernel_size, stride, padding),
        inputs,
    )
    .into()
}

pub trait AvgPool1dOp {
    fn avg_pool1d<A: AvgPool1dArgs>(self, args: A) -> RaiResult<Tensor>;
}

impl<T> AvgPool1dOp for T
where
    T: TryAsTensor,
{
    #[inline]
    #[track_caller]
    fn avg_pool1d<A: AvgPool1dArgs>(self, args: A) -> RaiResult<Tensor> {
        avg_pool1d(self, args)
    }
}
