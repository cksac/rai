use crate::{Dim, Shape, Tensor};

#[track_caller]
pub fn unsqueeze(x: &Tensor, d: impl Dim) -> Tensor {
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
