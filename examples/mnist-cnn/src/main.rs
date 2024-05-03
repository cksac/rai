use rai::{
    device, eval,
    nn::{Conv2d, Conv2dConfig, Dropout, Linear, Module, TrainableModule},
    opt::{
        losses::softmax_cross_entropy_with_integer_labels,
        optimizers::{Optimizer, SDG},
    },
    value_and_grad, AsDevice, Aux, Device, Func, Module, Shape, Tensor, Type, F32,
};
use rai_datasets::image::mnist;
use rand::{seq::SliceRandom, thread_rng};
use std::{fmt::Debug, time::Instant};

#[derive(Debug, Clone, Module)]
#[module(input = (Tensor, bool))]
struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl ConvNet {
    pub fn new(num_classes: usize, dtype: impl Type, device: impl AsDevice) -> Self {
        let device = device.device();
        let conv1 = Conv2d::new(1, 32, 5, Conv2dConfig::default(), true, dtype, device);
        let conv2 = Conv2d::new(32, 64, 5, Conv2dConfig::default(), true, dtype, device);
        let fc1 = Linear::new(1024, 1024, true, dtype, device);
        let fc2 = Linear::new(1024, num_classes, true, dtype, device);
        let dropout = Dropout::new(0.5);
        Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        }
    }

    pub fn fwd(&self, xs: &Tensor, train: bool) -> Tensor {
        let b_sz = xs.shape_at(0);
        let xs = xs
            .reshape([b_sz, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d(2)
            .apply(&self.conv2)
            .max_pool2d(2)
            .flatten(1..)
            .apply(&self.fc1)
            .relu();
        self.dropout.fwd(&xs, train).apply(&self.fc2)
    }
}

fn loss_fn<M: TrainableModule<Input = (Tensor, bool), Output = Tensor>>(
    model: &M,
    input: &Tensor,
    train: bool,
    labels: &Tensor,
) -> (Tensor, Aux<Tensor>) {
    let logits = model.forward(&(input.clone(), train));
    let loss = softmax_cross_entropy_with_integer_labels(&logits, labels).mean(..);
    (loss, Aux(logits))
}

fn train_step<M: TrainableModule<Input = (Tensor, bool), Output = Tensor>, O: Optimizer>(
    optimizer: &mut O,
    model: &M,
    images: &Tensor,
    labels: &Tensor,
) -> (Tensor, Tensor) {
    let vg_fn = value_and_grad(loss_fn);
    let ((loss, Aux(logits)), (grads, ..)) = vg_fn.apply((model, images, true, labels));
    let mut params = optimizer.step(&grads);
    eval(&params);
    model.update_params(&mut params);
    (loss, logits)
}

fn main() {
    let num_classes = 10;
    let num_epochs = 10;
    let learning_rate = 0.05;
    let batch_size = 64;

    let device: Box<dyn Device> = device::cuda_if_available(0);
    let device = device.as_ref();
    println!("device: {:?}", device);
    let dtype = F32;

    let model = ConvNet::new(num_classes, dtype, device);
    let mut optimizer = SDG::new(model.params(), learning_rate);

    let dataset = mnist::load(device).expect("mnist dataset");
    let train_images = &dataset.train_images;
    let train_labels = &dataset.train_labels;
    let test_images = &dataset.test_images;
    let test_labels = &dataset.test_labels;
    let n_batches = train_images.shape_at(0) / batch_size;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    let start = Instant::now();
    for i in 0..num_epochs {
        let start = Instant::now();
        batch_idxs.shuffle(&mut thread_rng());
        let mut sum_loss = 0f32;
        for batch_idx in &batch_idxs {
            let train_images = train_images.narrow(0, batch_idx * batch_size, batch_size);
            let train_labels = train_labels.narrow(0, batch_idx * batch_size, batch_size);
            let (loss, _logits) = train_step(&mut optimizer, &model, &train_images, &train_labels);
            let loss = loss.as_scalar(F32);
            sum_loss += loss;
        }
        let avg_loss = sum_loss / n_batches as f32;
        let test_logits = model.fwd(test_images, false);
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
            avg_loss,
            test_accuracy * 100.0,
            elapsed,
        );
    }
    let elapsed = start.elapsed();
    let avg_elapsed = elapsed.as_secs_f64() / num_epochs as f64;
    println!("elapsed: {:?}, avg: {:.2} sec/epoch", elapsed, avg_elapsed);
    model.to_safetensors("mnist.safetensors");
}
