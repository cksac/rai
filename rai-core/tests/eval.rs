use rai_core::{backend::Cpu, DType, Tensor};

#[test]
fn test_add() {
    let backend = &Cpu;
    let a = &Tensor::ones([2, 3], DType::F32, backend);
    let b = &Tensor::ones([2, 3], DType::F32, backend);
    let c = &Tensor::ones([3], DType::F32, backend);

    let z = a + b + c;
    println!("{}", z);
}

#[test]
fn test_add2() {
    let backend = &Cpu;

    let a = &Tensor::ones([2, 3], DType::F32, backend);
    let b = &Tensor::full(1.4f32, [2, 3], backend);
    let c = &Tensor::full(1.4f32, [2, 3], backend);
    let d = &Tensor::full(1.4f32, [2, 3], backend);
    let e = &Tensor::full(1.4f32, [2, 3], backend);

    let z = a + (a * b - 3.2f32) - c / d + e;
    println!("{}", z);
}

#[test]
fn test_eval() {
    let backend = &Cpu;
    let a = &Tensor::full(1.0f32, [2, 3], backend);
    let b = &Tensor::full(2.0f32, [2, 3], backend);
    let c = &Tensor::full(3.0f32, [2, 3], backend);
    let d = &{
        let a = &Tensor::full(3.1f32, [2, 3], backend);
        let b = &Tensor::full(3.2f32, [2, 3], backend);
        a + b
    };
    println!("{}", d);

    let z = a / b + b * c - d;
    println!("{}", z);

    println!("{}", d);
}
