# RAI

![Rust](https://github.com/cksac/rai/workflows/Rust/badge.svg)
[![Docs Status](https://docs.rs/rai/badge.svg)](https://docs.rs/rai)
[![Latest Version](https://img.shields.io/crates/v/rai.svg)](https://crates.io/crates/rai)

ML framework with Ergonomic APIs in Rust. Lazy computation and composable transformations.
---
Note: It required `Rust nightly` with following features [`fn_traits`, `unboxed_closures`]

## Installation
```sh
cargo add rai
```

## Examples
### transformations (eval, grad, jvp, value_and_grad, vjp)
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

### linear regression
`cargo run --bin linear_regression --release`
```rust
use rai::{backend::Cpu, eval, grad, DType, Tensor};
use std::time::Instant;

fn main() {
    let num_features = 100;
    let num_samples = 1000;
    let num_iters = 1000;
    let learning_rate = 0.01f32;

    let backend = &Cpu;
    // True parameters
    let w_star = Tensor::normal([num_features], DType::F32, backend);

    // The input examples (design matrix)
    let x = Tensor::normal([num_samples, num_features], DType::F32, backend);

    // Noisy labels
    let eps = Tensor::normal([num_samples], DType::F32, backend) * 1e-2f32;
    let y = x.matmul(&w_star) + eps;

    // Initialize random parameters
    let w = &(Tensor::normal([num_features], DType::F32, backend) * 1e-2f32);

    let loss_fn = move |w: &Tensor| {
        let y = &y;
        let y_hat = x.matmul(&w);
        let loss = (y_hat - y).square().sum(..) * (0.5f32 / num_samples as f32);
        loss
    };

    let grad_fn = grad(loss_fn.clone());

    let start = Instant::now();
    for _ in 0..num_iters {
        let grads = grad_fn([w.clone()]);
        let grad = &grads[0];
        let new_w = w - grad * learning_rate;
        eval(&new_w);
        w.replace_data(new_w);
    }
    let elapsed = start.elapsed();
    let loss = loss_fn(w);
    let throughput = num_iters as f64 / elapsed.as_secs_f64();
    println!(
        "loss: {}, elapsed: {:?}, throughput: {:?} iters/sec",
        loss, elapsed, throughput
    );
}
```

### Neuron network modules with transformation (grad, jvp, value_and_grad, vjp)
```rust
#[test]
fn test_linear_grad() {
    let backend = &Cpu;

    let linear = Linear::new(5, 2, true, DType::F32, backend);
    let input = Tensor::normal([5], DType::F32, backend);

    let grad_fn = grad(linear);
    let grads = grad_fn(input);
    println!("{:?}", &grads);

    let grads = grads.tensors();
    println!("{}", grads[0]); // grad of linear.weight
    println!("{}", grads[1]); // grad of linear.bias
}
```

# LICENSE

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.
