use rai_core::{Shape, Tensor};

pub fn softmax_cross_entropy(logits: &Tensor, labels: &Tensor) -> Tensor {
    // todo: use log_softmax
    let t = labels * logits.softmax(-1);
    let dims = t.dims(..-1);
    -t.sum((dims, true))
}

pub fn softmax_cross_entropy_with_integer_labels(logits: &Tensor, labels: &Tensor) -> Tensor {
    todo!()
}
