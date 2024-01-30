use rai::{
    eval,
    nn::{Linear, Module, TrainableModule},
    opt::{
        losses::softmax_cross_entropy,
        optimizers::{Optimizer, SDG},
    },
    utils::cuda_enabled,
    value_and_grad, AsDevice, Aux, Cpu, Cuda, Device, Func, Module, Tensor, Type, F32,
};
use std::{collections::HashMap, fmt::Debug, time::Instant};

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
        layers.push(Linear::new(input_dim, hidden_dim, true, dtype, device));
        for _ in 1..num_layers - 2 {
            layers.push(Linear::new(hidden_dim, hidden_dim, true, dtype, device));
        }
        layers.push(Linear::new(hidden_dim, output_dim, true, dtype, device));
        Self { layers }
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
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
    let loss = softmax_cross_entropy(&logits, labels).mean(..);
    (loss, Aux(logits))
}

fn train_step<
    M: TrainableModule<
        Input = Tensor,
        Output = Tensor,
        Tensors = HashMap<usize, Tensor>,
        Gradient = HashMap<usize, Tensor>,
    >,
    O: Optimizer,
>(
    optimizer: &mut O,
    model: &M,
    input: &Tensor,
    labels: &Tensor,
) {
    let vg_fn = value_and_grad(loss_fn);
    let ((_loss, Aux(_logits)), (grads, ..)) = vg_fn.apply((model, input, labels));
    let mut params = optimizer.step(&grads);
    eval(&params);
    model.update_params(&mut params);
}

fn main() {
    let num_layers = 2;
    let hidden_dim = 32;
    let num_classes = 10;
    let batch_size = 256;
    let num_epochs = 10;
    let learning_rate = 1e-1;

    let device: &dyn Device = if cuda_enabled() { &Cuda(0) } else { &Cpu };
    let dtype = F32;

    let model = Mlp::new(num_layers, 784, hidden_dim, num_classes, dtype, device);
    let mut optimizer = SDG::new(model.params(), learning_rate);

    let start = Instant::now();
    for i in 0..num_epochs {
        let start = Instant::now();
        // todo: get image input and label
        let input = Tensor::rand([batch_size, 784], dtype, device);
        let labels = Tensor::full(0.123f32, [batch_size, 10], device);
        train_step(&mut optimizer, &model, &input, &labels);
        let elapsed = start.elapsed();
        println!("Epoch {i}: Time: {:?}", elapsed);
    }

    let elapsed = start.elapsed();
    let throughput = num_epochs as f64 / elapsed.as_secs_f64();
    println!(
        "elapsed: {:?}, throughput: {:.2} iters/sec",
        elapsed, throughput
    );

    model.to_safetensors("mnist.safetensors");

    // load saved model and test
    let loaded_model = Mlp::new(num_layers, 784, hidden_dim, num_classes, dtype, device);
    loaded_model.update_by_safetensors(&["mnist.safetensors"]);

    let input = Tensor::rand([batch_size, 784], dtype, device);
    let labels = Tensor::full(0.123f32, [batch_size, 10], device);
    let (loss, ..) = loss_fn(&loaded_model, &input, &labels);
    println!("loss = {}", loss);
}
