use rai_core::{
    eval, hessian, jacfwd, jacrev, utils::dot_graph, value_and_grad, Cpu, Shape, Tensor, F32,
};

#[test]
fn test_dot_graph() {
    let device = Cpu;
    let a = &Tensor::ones([2, 3], F32, device);
    let b = &Tensor::full(1.4f32, [2, 3], device);
    let c = &Tensor::full(1.4f32, [2, 3], device);
    let d = &Tensor::full(1.4f32, [2, 3], device);
    let e = &Tensor::full(1.4f32, [2, 3], device);
    let z = a + (a * b - 2.5f32) - c / d + e;
    println!("{}", z.dot_graph());
    println!("{}", z);
}

#[test]
fn test_arange() {
    let device = Cpu;
    let a1 = Tensor::arange(10.0f32, device);
    let a2 = Tensor::arange((10.0f32, 20.0f32), device);
    let a3 = Tensor::arange((10.0f32, 20.0f32, 2.0f32), device);
    let a4 = Tensor::arange((0u8, 10, 3), device);
    println!("{}", a1);
    println!("{}", a2);
    println!("{}", a3);
    println!("{}", a4);
}

#[test]
fn test_reshape() {
    let device = Cpu;
    let func = |x: &Tensor| x.reshape([6]);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_reshape_err() {
    let device = Cpu;
    let func = |x: &Tensor| x.reshape([6]);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 4], F32, device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_broadcast_to() {
    let device = Cpu;
    let func = |x: &Tensor| x.broadcast_to([3, 2, 3]);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_transpose() {
    let device = Cpu;
    let func = |x: &Tensor| x.transpose(-2, -1);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_matmul() {
    let device = Cpu;
    let func = |x: &Tensor, y: &Tensor| x.matmul(y);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, device);
    let b = Tensor::ones([3, 2], F32, device);
    let (out, (g1, g2)) = vg_func((&a, &b));
    println!("{}", dot_graph([&out, &g1, &g2]));
    println!("{}", out);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_matmul_2() {
    let device = Cpu;
    let func = |x: &Tensor, y: &Tensor| x.matmul(y);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, device);
    let b = Tensor::ones([3], F32, device);
    let (out, (g1, g2)) = vg_func((&a, &b));
    println!("{}", dot_graph([&out, &g1, &g2]));
    println!("{}", out);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_sum() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum(..);
    let vg_func = value_and_grad(func);
    let a = Tensor::full(2.3f32, [2, 3], device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_max() {
    let device = Cpu;
    let func = |x: &Tensor| x.max(0);
    let vg_func = value_and_grad(func);
    let a = Tensor::full(2.3f32, [2, 3], device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_min() {
    let device = Cpu;
    let func = |x: &Tensor| x.min(0);
    let vg_func = value_and_grad(func);
    let a = Tensor::full(2.3f32, [2, 3], device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_softmax() {
    let device = Cpu;
    let func = |x: &Tensor| x.softmax(1);
    let vg_func = value_and_grad(func);
    let a = Tensor::rand([2, 3], F32, device);
    let (out, grad) = vg_func(&a);
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_gather() {
    let device = Cpu;
    let a = Tensor::from_array([1u32, 2, 3, 4, 5, 6], [2, 3], device);
    let index = Tensor::from_array([0u32, 0, 0, 1, 1, 1], [2, 3], device);
    let out = a.gather(0, &index);
    println!("{}", a);
    println!("{}", index);
    println!("{}", out);
}

#[test]
fn test_index_select() {
    let device = Cpu;
    let a = Tensor::from_array([1u32, 2, 3, 4, 5, 6], [2, 3], device);
    let index = Tensor::from_array([1u32, 1, 0, 0], [4], device);
    let out = a.index_select(0, &index);
    println!("{}", a);
    println!("{}", index);
    println!("{}", out);
}

#[test]
fn test_flatten() {
    let device = Cpu;
    let a = Tensor::rand([2, 3, 4], F32, device);
    let out = a.flatten(..);
    println!("{}", a);
    println!("{}", out);
}

#[test]
fn test_conv1d() {
    let device = Cpu;
    let t = Tensor::rand([1, 4, 5], F32, device);
    let w = Tensor::rand([2, 4, 3], F32, device);
    let out = t.conv1d(w, 0, 1, 1, 1);
    println!("{}", out);
}

#[test]
fn test_conv1d_with_groups() {
    let device = Cpu;
    let t = Tensor::rand([1, 8, 5], F32, device);
    let w = Tensor::rand([2, 4, 3], F32, device);
    let out = t.conv1d(w, 0, 1, 1, 2);
    println!("{}", out);
}

#[test]
fn test_conv_transpose1d() {
    let device = Cpu;
    let t = Tensor::rand([1, 4, 5], F32, device);
    let w = Tensor::rand([2, 4, 3], F32, device).transpose(0, 1);
    let out = t.conv_transpose1d(w, 0, 0, 1, 1, 1);
    println!("{}", out);
}

#[test]
fn test_conv_transpose1d_with_groups() {
    let device = Cpu;
    let t = Tensor::rand([1, 4, 5], F32, device);
    let w = Tensor::rand([2, 4, 3], F32, device).transpose(0, 1);
    let out = t.conv_transpose1d(w, 0, 0, 1, 1, 2);
    println!("{}", out);
}

#[test]
fn test_conv2d() {
    let device = Cpu;
    let t = Tensor::rand([1, 4, 5, 5], F32, device);
    let w = Tensor::rand([2, 4, 3, 3], F32, device);
    let out = t.conv2d(w, [0, 0], [1, 1], [1, 1], 1);
    println!("{}", out);
}

#[test]
fn test_conv2d_with_groups() {
    let device = Cpu;
    let t = Tensor::rand([1, 4, 5, 5], F32, device);
    let w = Tensor::rand([2, 4, 3, 3], F32, device);
    let out = t.conv2d(w.transpose(0, 1), [0, 0], [1, 1], [1, 1], 2);
    println!("{}", out);
}

#[test]
fn test_conv_transpose2d() {
    let device = Cpu;
    let t = Tensor::rand([1, 4, 5, 5], F32, device);
    let w = Tensor::rand([2, 4, 3, 3], F32, device);
    let w_t = w.transpose(0, 1);
    let out = t.conv_transpose2d(w_t, [0, 0], [0, 0], [1, 1], [1, 1], 1);
    println!("{}", out);
}

#[test]
fn test_conv_transpose2d_with_groups() {
    let device = Cpu;
    let t = Tensor::rand([1, 4, 5, 5], F32, device);
    let w = Tensor::rand([2, 4, 3, 3], F32, device);
    let w_t = w.transpose(0, 1);
    let out = t.conv_transpose2d(w_t, [0, 0], [0, 0], [1, 1], [1, 1], 2);
    println!("{}", out);
}

#[test]
fn test_sign() {
    let device = Cpu;
    let func = |x: &Tensor| x.sign();
    let x = &Tensor::rand_with(-1.0f32, 1.0, [2, 2], device);
    let out = func(x);
    println!("{}", x);
    println!("{}", out);
}

#[test]
fn test_jacfwd() {
    let device = Cpu;
    let func = |x: &Tensor| x.sin();
    let jac_fn = jacfwd(func);
    let x = &Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], device);
    let out = jac_fn(x);
    eval((&x, &out)).unwrap();
    println!("{}", x);
    println!("{}", out);
    println!("{:?}", out.shape());
}

#[test]
fn test_jacrev() {
    let device = Cpu;
    let func = |x: &Tensor| x.sin();
    let jac_fn = jacrev(func);
    let x = &Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], device);
    let out = jac_fn(x);
    eval((&x, &out)).unwrap();
    println!("{}", x);
    println!("{}", out);
    println!("{:?}", out.shape());
}

#[test]
fn test_hessian() {
    let device = Cpu;
    let func = |x: &Tensor| x.sin();
    let hessian_fn = hessian(func);
    let x = &Tensor::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 4], device);
    let out = hessian_fn(x);
    eval((&x, &out)).unwrap();
    println!("{}", x);
    println!("{}", out);
    println!("{:?}", out.shape());
}
