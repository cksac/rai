use rai_core::{eval, jvp, linearize, utils::dot_graph, value_and_grad, Cpu, Func, Tensor, F32};

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
