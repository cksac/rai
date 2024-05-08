use rai::{eval, grad, Cpu, Tensor, F32};
use std::time::Instant;

fn main() {
    let num_features = 100;
    let num_samples = 1000;
    let num_iters = 10000;
    let learning_rate = 0.01f32;

    let device = Cpu;
    let dtype = F32;

    // True parameters
    let w_star = Tensor::randn([num_features], dtype, device);

    // The input examples (design matrix)
    let x = Tensor::randn([num_samples, num_features], dtype, device);

    // Noisy labels
    let eps = Tensor::randn([num_samples], dtype, device) * 1e-2f32;
    let y = x.matmul(&w_star) + eps;

    // Initialize random parameters
    let w = &(Tensor::randn([num_features], dtype, device) * 1e-2f32);

    let loss_fn = move |w: &Tensor| {
        let y = &y;
        let y_hat = x.matmul(w);
        (y_hat - y).square().sum(..) * (0.5f32 / num_samples as f32)
    };

    let grad_fn = grad(loss_fn.clone());

    let start = Instant::now();
    for _ in 0..num_iters {
        let grad = grad_fn(w);
        let new_w = w - grad * learning_rate;
        eval(&new_w);
        w.replace_data(new_w);
    }
    let elapsed = start.elapsed();
    let loss = loss_fn(w);
    let throughput = num_iters as f64 / elapsed.as_secs_f64();
    println!(
        "loss: {}, elapsed: {:?}, throughput: {:.2} iters/sec",
        loss, elapsed, throughput
    );
}
