use rai_core::{Shape, Tensor};

pub fn softmax_cross_entropy(logits: &Tensor, labels: &Tensor) -> Tensor {
    // todo: use log_softmax
    let t = labels * logits.softmax(logits.ndim() - 1);
    let mut axes = t.axes();
    axes.pop();
    -t.reduce_sum(axes)
}

pub fn softmax_cross_entropy_with_integer_labels(logits: &Tensor, labels: &Tensor) -> Tensor {
    todo!()
}
