# RAI
ML framework with Ergonomic APIs in Rust. Lazy computation and composable transformations.


# Examples
## linear regression
`cargo run --bin linear_regression --release`
```rust
use rai::{backend::Cpu, eval, grad, DType, Tensor};
use std::time::Instant;

fn main() {
    let num_features = 100;
    let num_samples = 1000;
    let num_iters = 1000;
    let learning_rate = 0.01f32;

    let backend = &Cpu::new();
    // True parameters
    let w_star = Tensor::normal([num_features], DType::F32, backend);

    // The input examples (design matrix)
    let x = Tensor::normal([num_samples, num_features], DType::F32, backend);

    // Noisy labels
    let eps = Tensor::normal([num_samples], DType::F32, backend) * 1e-2f32;
    let y = x.matmul(&w_star) + eps;

    // Initialize random parameters
    let mut w = Tensor::normal([num_features], DType::F32, backend) * 1e-2f32;

    let loss_fn = move |w: &Tensor| {
        let y = &y;
        let y_hat = x.matmul(&w);
        let loss = (y_hat - y).square().sum() * (0.5f32 / num_samples as f32);
        loss
    };

    let grad_fn = grad(loss_fn.clone());

    let start = Instant::now();
    for _ in 0..num_iters {
        let grads = grad_fn(&[w.clone()]);
        let grad = &grads[0];
        w = w - grad * learning_rate;
        eval(&w);
    }
    let elapsed = start.elapsed();
    let loss = loss_fn(&w);
    let throughput = num_iters as f64 / elapsed.as_secs_f64();
    println!(
        "loss: {}, elapsed: {:?}, throughput: {:?} iters/sec",
        loss, elapsed, throughput
    );
}
```

# LICENSE

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.
