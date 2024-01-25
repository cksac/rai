use rai_core::{Cpu, Tensor, F32};

#[test]
fn test_add() {
    let device = &Cpu;
    let a = &Tensor::ones([2, 3], F32, device);
    let b = &Tensor::ones([2, 3], F32, device);
    let c = &Tensor::ones([3], F32, device);

    let z = a + b + c;
    println!("{}", z);
}

#[test]
fn test_add2() {
    let device = &Cpu;

    let a = &Tensor::ones([2, 3], F32, device);
    let b = &Tensor::full(1.4f32, [2, 3], device);
    let c = &Tensor::full(1.4f32, [2, 3], device);
    let d = &Tensor::full(1.4f32, [2, 3], device);
    let e = &Tensor::full(1.4f32, [2, 3], device);

    let z = a + (a * b - 3.2f32) - c / d + e;
    println!("{}", z);
}

#[test]
fn test_eval() {
    let device = &Cpu;
    let a = &Tensor::full(1.0f32, [2, 3], device);
    let b = &Tensor::full(2.0f32, [2, 3], device);
    let c = &Tensor::full(3.0f32, [2, 3], device);
    let d = &{
        let a = &Tensor::full(3.1f32, [2, 3], device);
        let b = &Tensor::full(3.2f32, [2, 3], device);
        a + b
    };
    println!("{}", d);

    let z = a / b + b * c - d;
    println!("{}", z);

    println!("{}", d);
}
