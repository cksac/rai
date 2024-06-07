use crate::{Dim, RaiResult, Shape, Tensor, TryAsTensor};

#[track_caller]
pub fn unsqueeze(x: impl TryAsTensor, d: impl Dim) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let is_negative = d.is_negative();
    let dim = x.dim(d);
    let mut shape = x.shape().to_vec();
    if is_negative {
        shape.insert(dim + 1, 1);
    } else {
        shape.insert(dim, 1);
    }
    x.reshape(shape)
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn unsqueeze(&self, d: impl Dim) -> RaiResult<Tensor> {
        unsqueeze(self, d)
    }
}
