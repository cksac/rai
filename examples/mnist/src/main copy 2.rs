use rai::{
    device, eval,
    nn::{Linear, Module, TrainableModule},
    opt::{
        losses::softmax_cross_entropy_with_integer_labels,
        optimizers::{Optimizer, SDG},
    },
    value_and_grad, AsDevice, Aux, Device, Module, Shape, Tensor, Type, F32,
};
use rai_datasets::image::mnist;
use std::{fmt::Debug, time::Instant};

#[derive(Debug, Clone, Module)]
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    pub fn new(
        num_layers: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let device = device.device();
        let mut layers = Vec::with_capacity(num_layers);
        if num_layers > 1 {
            layers.push(Linear::new(input_dim, hidden_dim, true, dtype, device));
            for _ in 1..num_layers - 2 {
                layers.push(Linear::new(hidden_dim, hidden_dim, true, dtype, device));
            }
            layers.push(Linear::new(hidden_dim, output_dim, true, dtype, device));
        } else {
            layers.push(Linear::new(input_dim, output_dim, true, dtype, device));
        }
        Self { layers }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for l in self.layers[0..self.layers.len() - 1].iter() {
            x = l.forward(&x).relu();
        }
        self.layers[self.layers.len() - 1].forward(&x)
    }
}

fn loss_fn<M: TrainableModule<Input = Tensor, Output = Tensor>>(
    model: &M,
    input: &Tensor,
    labels: &Tensor,
) -> (Tensor, Aux<Tensor>) {
    let logits = model.forward(input);
    let loss = softmax_cross_entropy_with_integer_labels(&logits, labels).mean(..);
    (loss, Aux(logits))
}

fn train_step<M: TrainableModule<Input = Tensor, Output = Tensor>, O: Optimizer>(
    optimizer: &mut O,
    model: &M,
    images: &Tensor,
    labels: &Tensor,
) -> (Tensor, Tensor) {
    let vg_fn = value_and_grad(loss_fn);
    let ((loss, Aux(logits)), (grads, ..)) = vg_fn((model, images, labels));
    let mut params = optimizer.step(&grads);
    eval(&params);
    model.update_params(&mut params);
    (loss, logits)
}

fn main() {
    let num_layers = 2;
    let hidden_dim = 100;
    let num_classes = 10;
    let num_epochs = 200;
    let learning_rate = 0.05;

    let device: Box<dyn Device> = device::cuda_if_available(0);
    let device = device.as_ref();
    println!("device: {:?}", device);
    let dtype = F32;

    let model = Mlp::new(num_layers, 784, hidden_dim, num_classes, dtype, device);
    let mut optimizer = SDG::new(model.params(), learning_rate);

    let dataset = mnist::load(device).expect("mnist dataset");
    let train_images = &dataset.train_images;
    let train_labels = &dataset.train_labels;
    let test_images = &dataset.test_images;
    let test_labels = &dataset.test_labels;

    let start = Instant::now();
    for i in 0..num_epochs {
        let start = Instant::now();
        let (loss, _logits) = train_step(&mut optimizer, &model, train_images, train_labels);
        let loss = loss.as_scalar(F32);
        let test_logits = model.forward(test_images);
        let sum_ok = test_logits
            .argmax(-1)
            .to_dtype(test_labels)
            .eq(test_labels)
            .to_dtype(F32)
            .sum(..)
            .as_scalar(F32);
        let test_accuracy = sum_ok / test_labels.size() as f32;
        let elapsed = start.elapsed();
        println!(
            "epoch: {i:04}, train loss: {:10.5}, test acc: {:5.2}%, time: {:?}",
            loss,
            test_accuracy * 100.0,
            elapsed,
        );
    }
    let elapsed = start.elapsed();
    let avg_elapsed = elapsed.as_secs_f64() / num_epochs as f64;
    println!("elapsed: {:?}, avg: {:.2} sec/epoch", elapsed, avg_elapsed);
    model.to_safetensors("mnist.safetensors");
}
