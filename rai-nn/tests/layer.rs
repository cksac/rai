use std::collections::BTreeMap;

use rai_core::{backend::Cpu, grad, jvp, value_and_grad, DType, Module, Tensor, WithTensors, utils::dot_graph, Aux};
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


fn loss_fn(model: &Linear, x: &Tensor) -> (Tensor, Aux<Tensor>) {
    let output = model.forward(x);
    let loss = output.sum(..);
    (loss, Aux(output))
}

#[test]
fn test_linear_batch_input() {
    let backend = &Cpu;
    let in_size = 5;
    let out_size = 2;
    let batch_size = 8;
    let linear = Linear::new(in_size,out_size, true, DType::F32, backend);
    let input = Tensor::normal([batch_size, in_size], DType::F32, backend);

    let vg_fn = value_and_grad(loss_fn);
    let ((loss, Aux(output)), grads) = vg_fn((&linear, &input));
    println!("loss = {:?}", &loss);
    println!("output = {:?}", &output);
    println!("grads = {:?}", &grads);

    let grads = grads.tensors();
    println!("{}", grads[0]);
    println!("{}", grads[1]);

    println!("{}", dot_graph((output, grads)));
}