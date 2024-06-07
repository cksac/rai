use rai_core::{Tensor, TensorOps};

pub fn l1_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    (predictions - targets).abs()
}

pub fn squared_error(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let errors = &(predictions - targets);
    // TODO: use integer power error.powi(2);
    errors * errors
}

pub fn l2_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    0.5 * squared_error(predictions, targets)
}
