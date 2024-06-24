use crate::{dim::Before, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow, fmt::Debug};

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

impl Op for MaxPool1d {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("MaxPool1d")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
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

    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for MaxPool1d")
    }

    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        assert_eq!(
            self.kernel_size, self.stride,
            "vjp not supported for maxpool1d if kernel_size != stride"
        );
        let x = &primals[0];
        let [_n, _c, l] = x.sizes(Before::<3>);
        let out_upsampled = &output.upsample_nearest1d(l);
        let mask = x.eq(out_upsampled).to_dtype(x);
        let avg = mask.avg_pool1d((self.kernel_size, self.stride));
        let cotan_x = (cotangent * avg).upsample_nearest1d(l) * mask;
        vec![cotan_x]
    }
}

pub trait MaxPool1dArgs: Debug {
    fn kernel_size(&self) -> usize;
    fn stride(&self) -> usize {
        self.kernel_size()
    }
    fn padding(&self) -> usize {
        0
    }
    fn dilation(&self) -> usize {
        1
    }
}

impl MaxPool1dArgs for usize {
    fn kernel_size(&self) -> usize {
        *self
    }
}

impl MaxPool1dArgs for (usize, usize) {
    fn kernel_size(&self) -> usize {
        self.0
    }

    fn stride(&self) -> usize {
        self.1
    }
}

impl MaxPool1dArgs for (usize, usize, usize) {
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

impl MaxPool1dArgs for (usize, usize, usize, usize) {
    fn kernel_size(&self) -> usize {
        self.0
    }

    fn stride(&self) -> usize {
        self.1
    }

    fn padding(&self) -> usize {
        self.2
    }

    fn dilation(&self) -> usize {
        self.3
    }
}

#[track_caller]
pub fn max_pool1d(input: &Tensor, args: impl MaxPool1dArgs) -> Tensor {
    let kernel_size = args.kernel_size();
    let stride = args.stride();
    let padding = args.padding();
    let dilation = args.dilation();
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_max_pool1d(kernel_size, stride, padding, dilation);
    let inputs = vec![input.clone()];
    Tensor::new(
        device,
        dtype,
        shape,
        MaxPool1d::new(kernel_size, stride, padding, dilation),
        inputs,
    )
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn max_pool1d(&self, args: impl MaxPool1dArgs) -> Tensor {
        max_pool1d(self, args)
    }
}
