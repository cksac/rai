use rai_core::{Shape, Tensor};

pub fn softmax_cross_entropy(logits: &Tensor, labels: &Tensor) -> Tensor {
    // todo: use log_softmax
    let t = labels * logits.softmax(logits.ndim() - 1);
    let dims = t.dims_until(-1);
    -t.reduce_sum(dims)
}

pub fn softmax_cross_entropy_with_integer_labels(logits: &Tensor, labels: &Tensor) -> Tensor {
    todo!()
}
