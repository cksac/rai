# RAI

![Rust](https://github.com/cksac/rai/workflows/Rust/badge.svg)
[![Docs Status](https://docs.rs/rai/badge.svg)](https://docs.rs/rai)
[![Latest Version](https://img.shields.io/crates/v/rai.svg)](https://crates.io/crates/rai)
[![Discord](https://img.shields.io/discord/1202429682474287144.svg?color=7289da&&logo=discord)](https://discord.gg/J7X8rNZeMC)

ML framework with ergonomic APIs in Rust. Lazy computation and composable transformations.
---

## Installation
```sh
cargo add rai
```

## Code snippets
### Function transformations (jvp, vjp, grad, value_and_grad)
```rust
use rai::{grad, Cpu, Func, Tensor, F32};

fn f(x: &Tensor) -> Tensor {
    x.sin()
}

fn main() {
    let grad_fn = grad(grad(f));
    let x = &Tensor::ones([1], F32, &Cpu);
    let grad = grad_fn.apply(x);
    println!("{}", grad.dot_graph());
    println!("{}", grad);
}
```

### NN Modules, Optimizer and loss functions
```rust
fn loss_fn<M: TrainableModule<Input = Tensor, Output = Tensor>>(
    model: &M,
    input: &Tensor,
    labels: &Tensor,
) -> (Tensor, Aux<Tensor>) {
    let logits = model.forward(input);
    let loss = softmax_cross_entropy(&logits, labels).mean(..);
    (loss, Aux(logits))
}

fn train_step<M: TrainableModule<Input = Tensor, Output = Tensor>, O: Optimizer>(
    optimizer: &mut O,
    model: &M,
    input: &Tensor,
    labels: &Tensor,
) {
    let vg_fn = value_and_grad(loss_fn);
    let ((_loss, Aux(_logits)), (grads, ..)) = vg_fn.apply((model, input, labels));
    let mut params = optimizer.step(&grads);
    eval(&params);
    model.update_params(&mut params);
}
```

## Examples
- [linear_regression](https://github.com/cksac/rai/blob/main/examples/linear_regression/src/main.rs)
    - `cargo run --bin linear_regression --release`
- [mnist](https://github.com/cksac/rai/blob/main/examples/mnist/src/main.rs)
    - `cargo run --bin mnist --release`
    - `cargo run --bin mnist --release --features=cuda`
- [mnist-cnn](https://github.com/cksac/rai/blob/main/examples/mnist-cnn/src/main.rs)
    - `cargo run --bin mnist-cnn --release`
    - `cargo run --bin mnist-cnn --release --features=cuda`
- [phi2](https://github.com/cksac/rai/blob/main/examples/phi2/src/main.rs)
    - `cargo run --bin phi2 --release`
- [qwen2](https://github.com/cksac/rai/blob/main/examples/qwen2/src/main.rs)
    - `cargo run --bin qwen2 --release`
- [gemma](https://github.com/cksac/rai/blob/main/examples/gemma/src/main.rs)
    - accept license agreement in https://huggingface.co/google/gemma-2b
    - `pip install huggingface_hub`
    - login to hf `huggingface-cli login`
    - `cargo run --bin gemma --release`
- [vit](https://github.com/cksac/rai/blob/main/examples/vit/src/main.rs)
    - `cargo run --bin vit --release`

## LICENSE
This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.
