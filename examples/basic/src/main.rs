use rai::{grad, Cpu, Tensor, F32};

fn f(x: &Tensor) -> Tensor {
    x.sin()
}

fn main() {
    let grad_fn = grad(grad(f));
    let x = &Tensor::ones([1], F32, Cpu);
    let grad = grad_fn(x);
    println!("{}", grad.dot_graph());
    println!("{}", grad);
}
