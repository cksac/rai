use rai::backend::Cpu;
use rai::opt::losses::softmax_cross_entropy;
use rai::utils::dot_graph;
use rai::{eval, WithTensors};
use rai::{nn::Linear, value_and_grad, Backend, DType, Module, Tensor};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Mlp {
    ln1: Linear,
    ln2: Linear,
}

impl Mlp {
    pub fn new(
        image_dim: usize,
        label_dim: usize,
        dtype: DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Self {
        let backend = &backend.into();
        let ln1 = Linear::new(image_dim, 100, true, dtype, backend);
        let ln2 = Linear::new(100, label_dim, true, dtype, backend);
        Self { ln1, ln2 }
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = &self.ln1.forward(x);
        let x = &x.relu();
        self.ln2.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        [self.ln1.parameters(), self.ln2.parameters()].concat()
    }

    fn update(&self, params: &mut BTreeMap<usize, Tensor>) {
        self.ln1.update(params);
        self.ln2.update(params);
    }
}

fn loss_fn(model: &Mlp, input: &Tensor, labels: &Tensor) -> (Tensor, Tensor) {
    // TODO: return correct loss
    let logits = model.forward(input);
    let loss = softmax_cross_entropy(&logits, labels);
    dbg!(&logits, &loss);
    (loss, logits)
}

fn train(model: &Mlp, input: &Tensor, label: &Tensor) {
    let vg_fn = value_and_grad(loss_fn);
    let ((_loss, _logits), grads) = vg_fn((model, input, label));

    println!("{:?}", model.parameters());
    println!("{:?}", grads);

    // TODO: Optimizer to get new params
    let mut new_params: BTreeMap<usize, Tensor> = model
        .parameters()
        .iter()
        .map(|t| {
            let id = t.id();
            let grad = grads.get(&id).unwrap();
            let new_t = t - grad * 0.01;
            dbg!(id, t, grad, &new_t);
            println!("{}", dot_graph([t, grad, &new_t]));
            (id, new_t)
        })
        .collect();
    eval(new_params.tensors());

    // apply param update
    model.update(&mut new_params);
}

fn main() {
    let num_iters = 1;
    let batch_size = 5;
    let backend = &Cpu;
    let dtype = DType::F32;

    let model = Mlp::new(784, 10, dtype, backend);

    let start = Instant::now();
    for _ in 0..num_iters {
        // todo: get image input and label
        let input = Tensor::normal([batch_size, 784], dtype, backend);
        let labels = Tensor::full(0.123, [batch_size, 10], dtype, backend);
        train(&model, &input, &labels);
    }
    let elapsed = start.elapsed();
    let throughput = num_iters as f64 / elapsed.as_secs_f64();
    println!(
        "elapsed: {:?}, throughput: {:?} iters/sec",
        elapsed, throughput
    );
    // todo: save model
}
