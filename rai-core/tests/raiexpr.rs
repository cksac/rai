use rai_core::{raiexpr, value_and_grad, Cpu, Tensor, F32};

#[test]
fn test_linear_grad_expr() {
    let device = Cpu;

    // need explicit type annotations
    let func = |w: &Tensor, b: &Tensor, x: &Tensor| (x.matmul(w.t()) + b).sum(..);
    let vg_func = value_and_grad(func);

    let in_dim = 5;
    let out_dim = 2;
    let batch_dim = 8;

    let w = Tensor::ones([out_dim, in_dim], F32, device);
    let b = Tensor::ones([out_dim], F32, device);
    let x = Tensor::ones([batch_dim, in_dim], F32, device);

    let e = raiexpr(&vg_func, (&w, &b, &x)).unwrap();
    println!("{}", e);
}
