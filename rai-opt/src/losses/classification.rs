use rai_core::{Shape, Tensor};

pub fn softmax_cross_entropy(logits: &Tensor, labels: &Tensor) -> Tensor {
    let t = labels * logits.log_softmax(-1);
    -t.sum((..-1, true))
}

pub fn softmax_cross_entropy_with_integer_labels(logits: &Tensor, labels: &Tensor) -> Tensor {
    let logits_max = logits.max((-1, true));
    let logits = logits - logits_max;
    let idx = labels.broadcast_right(logits.shape_of([-1]));
    let label_logits = logits.gather(-1, idx).narrow(-1, 0, 1).squeeze(-1);
    let log_normalizers = logits.exp().sum(-1).log();
    log_normalizers - label_logits
}
