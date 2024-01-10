use rai_core::backend::Cpu;
use rai_core::eval;
use rai_core::utils::dot_graph;
use rai_core::{value_and_grad, DType::F32, Tensor};

#[test]
fn test_linear_grad() {
    let backend = &Cpu;

    // need explicit type annotations
    let func = |w: &Tensor, b: &Tensor, x: &Tensor| (x.matmul(w.t()) + b).sum(..);
    let vg_func = value_and_grad(func);

    let in_dim = 5;
    let out_dim = 2;
    let batch_dim = 8;

    let w = Tensor::ones([out_dim, in_dim], F32, backend);
    let b = Tensor::ones([out_dim], F32, backend);
    let x = Tensor::ones([batch_dim, in_dim], F32, backend);

    let (v, g) = vg_func([w, b, x]);
    eval(((&v, &g), true));
    println!("output = {}", v[0]);
    println!("grad_w = {}", g[0]);
    println!("grad_b = {}", g[1]);
    println!("{}", dot_graph((&v, &g)))
}
