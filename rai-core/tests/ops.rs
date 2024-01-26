use rai_core::{utils::dot_graph, value_and_grad, Cpu, Func, Tensor, F32};

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
    let (out, grad) = vg_func.apply((&a,));
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
    let (out, grad) = vg_func.apply((&a,));
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
    let (out, grad) = vg_func.apply((&a,));
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
    let (out, (g1, g2)) = vg_func.apply((&a, &b));
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
    let (out, (g1, g2)) = vg_func.apply((&a, &b));
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
    let (out, grad) = vg_func.apply((&a,));
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
    let (out, grad) = vg_func.apply((&a,));
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
    let (out, grad) = vg_func.apply((&a,));
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
    let (out, grad) = vg_func.apply((&a,));
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
