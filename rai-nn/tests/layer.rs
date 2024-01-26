use rai_core::{nn::Module, utils::dot_graph, value_and_grad, Aux, Cpu, Func, Tensor, F32};

use rai_nn::{Embedding, Linear};

fn loss_fn(model: &Linear, x: &Tensor) -> (Tensor, Aux<Tensor>) {
    let output = model.forward(x);
    let loss = output.sum(..);
    (loss, Aux(output))
}

#[test]
fn test_linear_batch_input() {
    let device = Cpu;
    let in_size = 5;
    let out_size = 2;
    let batch_size = 8;
    let linear = Linear::new(in_size, out_size, true, F32, device);
    let input = Tensor::randn([batch_size, in_size], F32, device);

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

#[test]
fn test_embedding() {
    let device = Cpu;
    let num_embeddings = 10;
    let features = 4;
    let embedding = Embedding::new(num_embeddings, features, F32, device);
    let input = Tensor::from_array([0u32, 1, 2, 3, 4], [5], device);

    let output = embedding.forward(&input);
    println!("embeddings = {}", embedding.weight());
    println!("output = {}", &output);
}
