use crate::{Dims, RaiResult, Shape, Tensor, TryAsTensor};

#[track_caller]
pub fn squeeze(x: impl TryAsTensor, dims: impl Dims<Vec<usize>>) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let dims = x.dims(dims);
    let mut out_shape = Vec::new();
    for (i, s) in x.shape().iter().enumerate() {
        if !dims.contains(&i) || *s != 1 {
            out_shape.push(*s);
        }
    }
    x.reshape(out_shape)
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn squeeze(&self, dims: impl Dims<Vec<usize>>) -> RaiResult<Tensor> {
        squeeze(self, dims)
    }
}
