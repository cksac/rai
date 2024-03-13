use rai_core::Tensor;

pub fn softmax_cross_entropy(logits: &Tensor, labels: &Tensor) -> Tensor {
    let t = labels * logits.log_softmax(-1);
    -t.sum((..-1, true))
}

pub fn softmax_cross_entropy_with_integer_labels(logits: &Tensor, labels: &Tensor) -> Tensor {
    -logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1))
}
