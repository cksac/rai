use rai_core::{
    backend::{Cpu, RaiExpr},
    eval, value_and_grad,
    DType::F32,
};
use rai_core::{raiexpr, Tensor};

#[test]
fn test_add_expr() {
    let backend = &Cpu;
    let a = &Tensor::ones([2, 3], F32, backend);
    let b = &Tensor::ones([2, 3], F32, backend);
    let c = &Tensor::ones([3], F32, backend);

    let z = a + b + c;
    eval((&z, true, RaiExpr));
    println!("{}", z);
}

#[test]
fn test_linear_grad_expr() {
    let backend = &Cpu;

    // need explicit type annotations
    let func = |w: &Tensor, b: &Tensor, x: &Tensor| (x.matmul(w.t()) + b).sum(..);
    let vg_func = value_and_grad(func);
    let ir_func = raiexpr(vg_func);

    let in_dim = 5;
    let out_dim = 2;
    let batch_dim = 8;

    let w = Tensor::ones([out_dim, in_dim], F32, backend);
    let b = Tensor::ones([out_dim], F32, backend);
    let x = Tensor::ones([batch_dim, in_dim], F32, backend);

    let exprs = ir_func.raiexpr_of((&w, &b, &x));
    println!("{}", exprs);
}
