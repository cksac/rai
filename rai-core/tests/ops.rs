use rai_core::{backend::Cpu, utils::dot_graph, value_and_grad, DType, Shape, Tensor};

#[test]
fn test_dot_graph() {
    let backend = &Cpu;

    let a = &Tensor::ones([2, 3], DType::F32, backend);
    let b = &Tensor::full(1.4, [2, 3], DType::F32, backend);
    let c = &Tensor::full(1.4, [2, 3], DType::F32, backend);
    let d = &Tensor::full(1.4, [2, 3], DType::F32, backend);
    let e = &Tensor::full(1.4, [2, 3], DType::F32, backend);

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
    let a4 = Tensor::arange((10.0f32, 20.0f32, 2.0f32, DType::F64), backend);
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
    let a = Tensor::ones([2, 3], DType::F32, backend);
    let (outs, grads) = vg_func([a]);
    println!("{}", dot_graph([&outs, &grads]));
    println!("{}", outs[0]);
    println!("{}", grads[0]);
}

#[test]
fn test_broadcast_to() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.broadcast_to([3, 2, 3]);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], DType::F32, backend);
    let (outs, grads) = vg_func([a]);
    println!("{}", dot_graph([&outs, &grads]));
    println!("{}", outs[0]);
    println!("{}", grads[0]);
}

#[test]
fn test_transpose() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.t();
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], DType::F32, backend);
    let (outs, grads) = vg_func([a]);
    println!("{}", dot_graph([&outs, &grads]));
    println!("{}", outs[0]);
    println!("{}", grads[0]);
}

#[test]
fn test_matmul() {
    let backend = &Cpu;
    let func = |x: &Tensor, y: &Tensor| x.matmul(y);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], DType::F32, backend);
    let b = Tensor::ones([3, 2], DType::F32, backend);
    let (outs, grads) = vg_func([a, b]);
    println!("{}", dot_graph((&outs, &grads)));
    println!("{}", outs[0]);
    println!("{}", grads[0]);
    println!("{}", grads[1]);
}

#[test]
fn test_matmul_2() {
    let backend = &Cpu;
    let func = |x: &Tensor, y: &Tensor| x.matmul(y);
    let vg_func = value_and_grad(func);
    let a = Tensor::ones([2, 3], DType::F32, backend);
    let b = Tensor::ones([3], DType::F32, backend);
    let (outs, grads) = vg_func([a, b]);
    println!("{}", dot_graph((&outs, &grads)));
    println!("{}", outs[0]);
    println!("{}", grads[0]);
    println!("{}", grads[1]);
}

#[test]
fn test_sum() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.sum();
    let vg_func = value_and_grad(func);
    let a = Tensor::full(2.3, [2, 3], DType::F32, backend);
    let (outs, grads) = vg_func([a]);
    println!("{}", dot_graph([&outs, &grads]));
    println!("{}", outs[0]);
    println!("{}", grads[0]);
}

#[test]
fn test_softmax() {
    let backend = &Cpu;
    let func = |x: &Tensor| x.softmax(-1);
    let vg_func = value_and_grad(func);

    let a = &Tensor::normal([2, 3], DType::F32, backend);
    let (outs, grads) = vg_func([a.clone()]);
    println!("{}", dot_graph([&outs, &grads]));

    println!("{:?}", outs[0]);
    println!("{:?}", grads[0]);

    println!("{}", a);
    println!("{}", outs[0]);
    println!("sum {}", outs[0].reduce_sum([-1]));
    println!("{}", grads[0]);
}
