use rai::{nn::Linear, value_and_grad, Module, Tensor};

#[derive(Debug, Clone)]
struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Module for Mlp {
    fn forward(&self, x: Tensor) -> Tensor {
        let x = self.ln1.forward(x);
        self.ln2.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        [self.ln1.parameters(), self.ln2.parameters()].concat()
    }
}

fn cross_entropy(logits: &Tensor, label: &Tensor) -> Tensor {
    //todo!()
    let loss = logits - label;
    loss
}

fn train_step(model: &mut Mlp, input: Tensor, label: Tensor) {
    let loss_fn = move |m: &Mlp, x: &Tensor| {
        let logits = m.forward(x.clone());
        let loss = cross_entropy(x, &label);
        (loss, logits)
    };
    let vg_fn = value_and_grad(loss_fn);
    let ((loss, logits), grads) = vg_fn((model, input));
}

fn main() {
    let epochs = 10;

    println!("Hello, world!");
}
