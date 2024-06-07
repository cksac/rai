use crate::{dim::Before, Op, RaiResult, Shape, Tensor, TryAsTensor};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq)]
pub struct UpsampleNearest1d {
    pub size: usize,
}

impl UpsampleNearest1d {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Op for UpsampleNearest1d {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("UpsampleNearest1d({})", self.size)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        todo!("jvp for UpsampleNearest1d")
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let [_n, c, size] = x.sizes(Before::<3>);
        assert!(
            self.size % size != 0,
            "UpsampleNearest1d vjp not supported for non integer upscaling factors"
        );
        let scale = self.size / size;
        let kernel = &Tensor::ones([c, 1, scale], x.dtype(), x.device());
        let cotan_x = cotangent.conv1d(kernel, 0, scale, 1, c);
        vec![cotan_x]
    }
}

#[track_caller]
pub fn upsample_nearest1d(input: impl TryAsTensor, size: usize) -> RaiResult<Tensor> {
    let input = crate::try_get! { input.try_as_tensor() };
    let device = input.device();
    let dtype = input.dtype();
    let shape = input.shape_upsample_nearest1d(size);
    let inputs = vec![input.clone()];
    Tensor::new(device, dtype, shape, UpsampleNearest1d::new(size), inputs).into()
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn upsample_nearest1d(&self, size: usize) -> RaiResult<Tensor> {
        upsample_nearest1d(self, size)
    }
}
