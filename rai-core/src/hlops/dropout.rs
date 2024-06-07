use crate::{RaiResult, Tensor, TryAsTensor};

#[track_caller]
pub fn dropout(input: impl TryAsTensor, p: f32) -> RaiResult<Tensor> {
    let input = crate::try_get! { input.try_as_tensor() };
    assert!((0.0..1.0).contains(&p));
    let r = crate::try_get! { input.rand_like() };
    let scale = 1.0 / (1.0 - p);
    let mask = r.ge(r.full_like(p)).to_dtype(r) * scale;
    input * mask
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn dropout(&self, p: f32) -> RaiResult<Tensor> {
        dropout(self, p)
    }
}
