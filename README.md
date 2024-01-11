# RAI

![Rust](https://github.com/cksac/rai/workflows/Rust/badge.svg)
[![Docs Status](https://docs.rs/rai/badge.svg)](https://docs.rs/rai)
[![Latest Version](https://img.shields.io/crates/v/rai.svg)](https://crates.io/crates/rai)

ML framework with Ergonomic APIs in Rust. Lazy computation and composable transformations.
---

## Installation
```sh
cargo add rai
```

## Code snippets
### Function transformations (jvp, vjp, grad, value_and_grad)
```rust
use rai::backend::Cpu;
use rai::{grad, DType, Tensor};

fn f(x: &Tensor) -> Tensor {
    x.sin()
}

fn main() {
    let grad_fn = grad(grad(f));

    let backend = &Cpu;
    let x = Tensor::ones([1], DType::F32, backend);
    let grads = grad_fn([x]);

    println!("{}", grads[0].dot_graph());
    println!("{}", grads[0]);
}
```

### NN Modules, Optimizer and loss functions
```rust
fn loss_fn<M: Module + 'static>(
    model: &M,
    input: &Tensor,
    labels: &Tensor,
) -> (Tensor, Aux<Tensor>) {
    let logits = model.forward(input);
    let loss = softmax_cross_entropy(&logits, labels).mean(..);
    (loss, Aux(logits))
}

fn train_step<O: Optimizer, M: Module + 'static>(
    optimizer: &mut O,
    model: &M,
    input: &Tensor,
    labels: &Tensor,
) {
    let vg_fn = value_and_grad(loss_fn);
    let ((_loss, Aux(_logits)), grads) = vg_fn((model, input, labels));
    let mut params = optimizer.step(&grads);
    eval(&params);
    model.update(&mut params);
}
```

## Examples
- [linear_regression](https://github.com/cksac/rai/blob/main/examples/linear_regression/src/main.rs)
    - `cargo run --bin linear_regression --release`
- [mnist](https://github.com/cksac/rai/blob/main/examples/mnist/src/main.rs)
    - `cargo run --bin mnist --release`



# LICENSE

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.
