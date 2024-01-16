use rai_core::{backend::Cpu, utils::dot_graph, value_and_grad, Func, Tensor, F32};

#[test]
fn test_dot_graph() {
    let backend = &Cpu;

    let a = &Tensor::ones([2, 3], F32, backend);
    let b = &Tensor::full(1.4f32, [2, 3], backend);
    let c = &Tensor::full(1.4f32, [2, 3], backend);
    let d = &Tensor::full(1.4f32, [2, 3], backend);
    let e = &Tensor::full(1.4f32, [2, 3], backend);

    let z = a + (a * b - 2.5f32) - c / d + e;
    println!("{}", z.dot_graph());
    println!("{}", z);
}

#[test]
fn test_arange() {
    let backend = &Cpu;
    let a1 = Tensor::arange(10.0f32, backend);
    let a2 = Tensor::arange((10.0f32, 20.0f32), backend);
    let a3 = Tensor::arange((10.0f32, 20.0f32, 2.0f32), backend);
    let a4 = Tensor::arange((0u8, 10, 3), backend);
    println!("{}", a1);
    println!("{}", a2);
    println!("{}", a3);
    println!("{}", a4);
}

#[test]
fn test_reshape() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.reshape([6]);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, backend);
    let (out, grad) = vg_func.apply((&a,));
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_broadcast_to() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.broadcast_to([3, 2, 3]);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, backend);
    let (out, grad) = vg_func.apply((&a,));
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_transpose() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.t();
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, backend);
    let (out, grad) = vg_func.apply((&a,));
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_matmul() {
    let backend = &Cpu;
    let func = |x: &Tensor, y: &Tensor| x.matmul(y);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, backend);
    let b = Tensor::ones([3, 2], F32, backend);
    let (out, (g1, g2)) = vg_func.apply((&a, &b));
    println!("{}", dot_graph([&out, &g1, &g2]));
    println!("{}", out);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_matmul_2() {
    let backend = &Cpu;
    let func = |x: &Tensor, y: &Tensor| x.matmul(y);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], F32, backend);
    let b = Tensor::ones([3], F32, backend);
    let (out, (g1, g2)) = vg_func.apply((&a, &b));
    println!("{}", dot_graph([&out, &g1, &g2]));
    println!("{}", out);
    println!("{}", g1);
    println!("{}", g2);
}

#[test]
fn test_sum() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.sum(..);
    let vg_func = value_and_grad(func);
    let a = Tensor::full(2.3f32, [2, 3], backend);
    let (out, grad) = vg_func.apply((&a,));
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_max() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.max(0);
    let vg_func = value_and_grad(func);
    let a = Tensor::full(2.3f32, [2, 3], backend);
    let (out, grad) = vg_func.apply((&a,));
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_min() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.min(0);
    let vg_func = value_and_grad(func);
    let a = Tensor::full(2.3f32, [2, 3], backend);
    let (out, grad) = vg_func.apply((&a,));
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_softmax() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.softmax(1);
    let vg_func = value_and_grad(func);

    let a = Tensor::normal([2, 3], F32, backend);
    let (out, grad) = vg_func.apply((&a,));
    println!("{}", dot_graph([&out, &grad]));
    println!("{}", out);
    println!("{}", grad);
}

#[test]
fn test_gather() {
    let backend = &Cpu;

    let a = Tensor::from_array([1, 2, 3, 4, 5, 6], [2, 3], backend);
    let indexes = Tensor::from_array([0, 1, 1, 0, 0, 1], [2, 3], backend);
    let out = a.gather(0, &indexes);

    println!("{}", a);
    println!("{}", indexes);
    println!("{}", out);
}
