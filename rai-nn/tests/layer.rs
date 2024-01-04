use std::collections::BTreeMap;

use rai::{backend::Cpu, grad, jvp, value_and_grad, DType, Module, Tensor, WithTensors};
use rai_nn::Linear;

#[test]
fn test_linear_jvp() {
    let backend = &Cpu;

    let linear = Linear::new(100, 10, true, DType::F32, backend);
    let input = Tensor::normal([100], DType::F32, backend);

    let tangents: BTreeMap<usize, Tensor> = linear
        .parameters()
        .iter()
        .map(|t| (t.id(), t.ones_like()))
        .collect();

    let (output, jvps) = jvp(linear, input, tangents);
    println!("{}", output);
    println!("{}", jvps.get(&output.id()).unwrap());
}

#[test]
fn test_linear_grad() {
    let backend = &Cpu;

    let linear = Linear::new(5, 2, true, DType::F32, backend);
    let input = Tensor::normal([5], DType::F32, backend);

    let grad_fn = grad(linear);
    let grads = grad_fn(input);
    println!("{:?}", &grads);

    let grads = grads.tensors();
    println!("{}", grads[0]);
    println!("{}", grads[1]);
}

#[test]
fn test_linear_grad_of_grad() {
    let backend = &Cpu;

    let linear = Linear::new(5, 2, true, DType::F32, backend);
    let input = Tensor::normal([5], DType::F32, backend);

    let grad_fn = grad(grad(linear));
    let grads = grad_fn(input);
    let grads = grads.tensors();

    println!("{}", grads[0]);
    println!("{}", grads[1]);
}

#[test]
fn test_linear_value_and_grad_of_grad() {
    let backend = &Cpu;

    let linear = Linear::new(5, 2, true, DType::F32, backend);
    let input = Tensor::normal([5], DType::F32, backend);

    let grad_fn = value_and_grad(grad(linear));
    let (output, grads) = grad_fn(input);
    let grads = grads.tensors();

    println!("{}", output);
    println!("{}", grads[0]);
    println!("{}", grads[1]);
}
