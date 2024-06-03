use crate::{Dims, Shape, Tensor};

#[track_caller]
pub fn squeeze(x: &Tensor, dims: impl Dims) -> Tensor {
    let dims = x.dims(dims).to_vec();
    let mut out_shape = Vec::new();
    for (i, s) in x.shape().iter().enumerate() {
        if !dims.contains(&i) || *s != 1 {
            out_shape.push(*s);
        }
    }
    x.reshape(out_shape)
}
