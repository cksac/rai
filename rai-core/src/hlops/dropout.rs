use crate::Tensor;

#[track_caller]
pub fn dropout(input: &Tensor, p: f32) -> Tensor {
    assert!((0.0..1.0).contains(&p));
    let r = input.rand_like();
    let scale = 1.0 / (1.0 - p);
    let mask = r.ge(r.full_like(p)).to_dtype(r) * scale;
    input * mask
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn dropout(&self, p: f32) -> Tensor {
        dropout(self, p)
    }
}
