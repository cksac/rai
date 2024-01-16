use rai::backend::Cpu;
use rai::opt::losses::softmax_cross_entropy;
use rai::opt::optimizers::{Optimizer, SDG};
use rai::{differentiable_module, eval, Aux, DType, DifferentiableModule, F32};
use rai::{nn::Linear, value_and_grad, Backend, Func, Module, Tensor};
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    pub fn new(
        num_layers: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        dtype: impl DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Self {
        let backend = &backend.into();
        let mut layers = Vec::with_capacity(num_layers);
        layers.push(Linear::new(input_dim, hidden_dim, true, dtype, backend));
        for _ in 1..num_layers - 2 {
            layers.push(Linear::new(hidden_dim, hidden_dim, true, dtype, backend));
        }
        layers.push(Linear::new(hidden_dim, output_dim, true, dtype, backend));
        Self { layers }
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.clone();
        for l in self.layers[0..self.layers.len() - 1].iter() {
            x = l.forward(&x).relu();
        }
        self.layers[self.layers.len() - 1].forward(&x)
    }

    fn gather_params(&self, out: &mut HashMap<usize, Tensor>) {
        for l in &self.layers {
            l.gather_params(out)
        }
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        for layer in self.layers.iter() {
            layer.update_params(params);
        }
    }
}
differentiable_module!(Mlp);

fn loss_fn<M: DifferentiableModule + 'static>(
    model: &M,
    input: &Tensor,
    labels: &Tensor,
) -> (Tensor, Aux<Tensor>) {
    let logits = model.forward(input);
    let loss = softmax_cross_entropy(&logits, labels).mean(..);
    (loss, Aux(logits))
}

fn train_step<O: Optimizer, M: DifferentiableModule + 'static>(
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

    let backend = &Cpu;
    let dtype = F32;

    let model = Mlp::new(num_layers, 784, hidden_dim, num_classes, dtype, backend);
    let mut optimizer = SDG::new(model.params(), learning_rate);

    let start = Instant::now();
    for i in 0..num_epochs {
        let start = Instant::now();
        // todo: get image input and label
        let input = Tensor::normal([batch_size, 784], dtype, backend);
        let labels = Tensor::full(0.123f32, [batch_size, 10], backend);
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
    // todo: save model
}
