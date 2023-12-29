use rai_core::{backend::Cpu, value_and_grad, DType, Tensor};

#[test]
fn test_add_grad() {
    let backend = &Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x + y;
    let vg_func = value_and_grad(func);

    let a = Tensor::ones([1], DType::F32, backend);
    let b = Tensor::ones([1], DType::F32, backend);

    let (v, g) = vg_func(&[a, b]);
    println!("{}", v[0]);
    println!("{}", g[0]);
    println!("{}", g[1]);
}

#[test]
fn test_sub_grad() {
    let backend = &Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x - y;
    let vg_func = value_and_grad(func);

    let a = Tensor::ones([1], DType::F32, backend);
    let b = Tensor::ones([1], DType::F32, backend);

    let (v, g) = vg_func(&[a, b]);
    println!("{}", v[0]);
    println!("{}", g[0]);
    println!("{}", g[1]);
}

#[test]
fn test_mul_grad() {
    let backend = &Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x * y;
    let vg_func = value_and_grad(func);

    let a = Tensor::full(3.0, [1], DType::F32, backend);
    let b = Tensor::full(4.0, [1], DType::F32, backend);

    let (v, g) = vg_func(&[a, b]);
    println!("{}", v[0]);
    println!("{}", g[0]);
    println!("{}", g[1]);
}

#[test]
fn test_div_grad() {
    let backend = &Cpu;

    // need explicit type annotations
    let func = |x: &Tensor, y: &Tensor| x / y;
    let vg_func = value_and_grad(func);

    let a = Tensor::ones([1], DType::F32, backend);
    let b = Tensor::ones([1], DType::F32, backend);

    let (v, g) = vg_func(&[a, b]);
    println!("{}", v[0]);
    println!("{}", g[0]);
    println!("{}", g[1]);
}
