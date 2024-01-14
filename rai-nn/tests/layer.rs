use rai_core::{
    backend::Cpu, utils::dot_graph, value_and_grad, Aux, DynDType, Func, Module, Tensor,
};

use rai_nn::Linear;

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
    let linear = Linear::new(in_size, out_size, true, DynDType::F32, backend);
    let input = Tensor::normal([batch_size, in_size], DynDType::F32, backend);

    let vg_fn = value_and_grad(loss_fn);
    let ((loss, Aux(output)), (grads, ..)) = vg_fn.apply((&linear, &input));
    println!("loss = {:?}", &loss);
    println!("output = {:?}", &output);
    println!("grads = {:?}", &grads);

    for (id, g) in grads.iter() {
        println!("grad of {id} = {g}")
    }

    println!("{}", dot_graph((output, grads)));
}
