use rai_core::{
    eval, jvp, linearize,
    utils::{check_vjp, dot_graph},
    value_and_grad, Cpu, Func, Tensor, F32,
};

#[test]
fn test_linearize() {
    let device = Cpu;
    let func = |x: &Tensor, y: &Tensor| x.sin() * y.cos();

    let a = Tensor::ones([1], F32, device);
    let b = Tensor::ones([1], F32, device);
    let (y1, y_dot1) = jvp(func, (&a, &b), (a.ones_like(), b.ones_like()));
    let (y2, f_lin) = linearize(func, (&a, &b));
    let y_dot2 = f_lin((a.ones_like(), b.ones_like()));
    println!("{}", y1);
    println!("{}", y2);
    println!("{}", y_dot1);
    println!("{}", y_dot2);
}

#[test]
fn test_add_grad() {
    let device = Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x + y;
    let vg_func = value_and_grad(func);

    let a = Tensor::ones([1], F32, device);
    let b = Tensor::ones([1], F32, device);

    let (v, (g1, g2)) = vg_func.apply((&a, &b));
    println!("{}", v);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_sub_grad() {
    let device = Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x - y;
    let vg_func = value_and_grad(func);

    let a = Tensor::ones([1], F32, device);
    let b = Tensor::ones([1], F32, device);

    let (v, (g1, g2)) = vg_func.apply((&a, &b));
    println!("{}", v);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_mul_grad() {
    let device = Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x * y;
    let vg_func = value_and_grad(func);

    let a = Tensor::full(3.0f32, [1], device);
    let b = Tensor::full(4.0f32, [1], device);

    let (v, (g1, g2)) = vg_func.apply((&a, &b));
    println!("{}", v);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_div_grad() {
    let device = Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x / y;
    let vg_func = value_and_grad(func);

    let a = Tensor::ones([1], F32, device);
    let b = Tensor::ones([1], F32, device);

    let (v, (g1, g2)) = vg_func.apply((&a, &b));
    println!("{}", v);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_linear_grad() {
    let device = Cpu;

    // need explicit type annotations
    let func = |w: &Tensor, b: &Tensor, x: &Tensor| (x.matmul(w.t()) + b).sum(..);
    let vg_func = value_and_grad(func);

    let in_dim = 5;
    let out_dim = 2;
    let batch_dim = 8;

    let w = &Tensor::ones([out_dim, in_dim], F32, device);
    let b = &Tensor::ones([out_dim], F32, device);
    let x = &Tensor::ones([batch_dim, in_dim], F32, device);

    let (v, (gw, gb, gx)) = vg_func.apply((w, b, x));
    eval(([&v, &gw, &gb, &gx], true));
    println!("output = {}", v);
    println!("grad_w = {}", gw);
    println!("grad_b = {}", gb);
    println!("grad_x = {}", gx);
    println!("{}", dot_graph([&v, &gw, &gb, &gx]))
}

#[test]
fn check_add_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x + 2.0;
    let x = &Tensor::rand([1], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_sub_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x - 2.0;
    let x = &Tensor::rand([1], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_mul_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x * 2.0;
    let x = &Tensor::rand([1], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_div_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x / 2.0;
    let x = &Tensor::rand([1], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_sin_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sin();
    let x = &Tensor::rand([1], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_cos_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.cos();
    let x = &Tensor::rand([1], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_sum_all_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum(..);
    let x = &Tensor::rand([2, 3], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_sum_all_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum((.., true));
    let x = &Tensor::rand([2, 3], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_sum_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum(-1);
    let x = &Tensor::rand([2, 3], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_sum_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum((-1, true));
    let x = &Tensor::rand([2, 3, 5], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_mean_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.mean((-1, true));
    let x = &Tensor::rand([2, 3, 5], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_max_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.max((-1, true));
    let x = &Tensor::rand([2, 3, 5], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_max_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.max(-1);
    let x = &Tensor::rand([2, 3, 5], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_min_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.min((-1, true));
    let x = &Tensor::rand([2, 3, 5], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_min_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.min(-1);
    let x = &Tensor::rand([2, 3, 5], F32, device);
    check_vjp(func, x, 1e-4);
}

#[test]
fn check_gather_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.gather(-1, Tensor::from_array([1u8, 2], [2, 1], device));
    let x = &Tensor::rand([2, 4], F32, device);
    check_vjp(func, x, 1e-4);
}
