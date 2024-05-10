use clap::{Parser, ValueEnum};

pub mod rai_mnist {
    use rai::{
        device, eval,
        nn::{Conv2d, Conv2dConfig, Dropout, Linear, Module, TrainableModule},
        opt::optimizers::{Optimizer, SDG},
        value_and_grad, AsDevice, Device, Module, Shape, Tensor, Type, F32,
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
        labels: &Tensor,
    ) -> Tensor {
        let logits = model.forward(&(input.clone(), true));
        logits
            .log_softmax(-1)
            .gather(-1, labels.unsqueeze(-1))
            .neg()
            .mean(..)
    }

    pub fn training(num_epochs: usize, learning_rate: f64, batch_size: usize, gpu_id: usize) {
        println!("rai mnist training...");
        let num_classes = 10;
        let device: Box<dyn Device> = device::cuda_if_available(gpu_id);
        let device = device.as_ref();
        println!("device: {:?}", device);
        let dtype = F32;

        let model = &ConvNet::new(num_classes, dtype, device);
        let optimizer = &mut SDG::new(model.params(), learning_rate);

        let dataset = mnist::load(device).expect("mnist dataset");
        let train_images = &dataset.train_images;
        let train_labels = &dataset.train_labels;
        let test_images = &dataset.test_images;
        let test_labels = &dataset.test_labels;

        let n_batches = train_images.shape_at(0) / batch_size;
        let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        let vg_fn = value_and_grad(loss_fn);
        let step_fn = |model, optimizer: &mut SDG, train_images, train_labels| {
            let (loss, (grads, ..)) = vg_fn((model, train_images, train_labels));
            let params = optimizer.step(&grads);
            (loss, params)
        };
        let step_fn = rai::optimize(step_fn);
        let start = Instant::now();
        for i in 0..num_epochs {
            let start = Instant::now();
            batch_idxs.shuffle(&mut thread_rng());
            let mut sum_loss = 0f32;
            for batch_idx in &batch_idxs {
                let train_images = &train_images.narrow(0, batch_idx * batch_size, batch_size);
                let train_labels = &train_labels.narrow(0, batch_idx * batch_size, batch_size);
                let (loss, mut params) = step_fn((model, optimizer, train_images, train_labels));
                eval(&params);
                model.update_params(&mut params);
                let loss = loss.as_scalar(F32);
                sum_loss += loss;
            }
            let avg_loss = sum_loss / n_batches as f32;
            let test_logits: Tensor = model.fwd(test_images, false);
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
    }
}

pub mod candle_mnist {
    use candle_core::{DType, Result, Tensor, D};
    use candle_nn::{ops, Conv2d, Linear, ModuleT, Optimizer, VarBuilder, VarMap};
    use rand::{seq::SliceRandom, thread_rng};
    use std::time::Instant;

    #[derive(Debug)]
    struct ConvNet {
        conv1: Conv2d,
        conv2: Conv2d,
        fc1: Linear,
        fc2: Linear,
        dropout: candle_nn::Dropout,
    }

    impl ConvNet {
        fn new(vs: VarBuilder, num_classes: usize) -> Result<Self> {
            let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
            let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
            let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
            let fc2 = candle_nn::linear(1024, num_classes, vs.pp("fc2"))?;
            let dropout = candle_nn::Dropout::new(0.5);
            Ok(Self {
                conv1,
                conv2,
                fc1,
                fc2,
                dropout,
            })
        }

        fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
            let (b_sz, _img_dim) = xs.dims2()?;
            let xs = xs
                .reshape((b_sz, 1, 28, 28))?
                .apply(&self.conv1)?
                .max_pool2d(2)?
                .apply(&self.conv2)?
                .max_pool2d(2)?
                .flatten_from(1)?
                .apply(&self.fc1)?
                .relu()?;
            self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
        }
    }

    fn loss_fn(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let log_sm = ops::log_softmax(logits, D::Minus1)?;
        let loss = log_sm
            .gather(&labels.unsqueeze(D::Minus1)?, D::Minus1)?
            .neg()?
            .mean_all()?;
        Ok(loss)
    }

    pub fn training(
        num_epochs: usize,
        learning_rate: f64,
        batch_size: usize,
        gpu_id: usize,
    ) -> anyhow::Result<()> {
        println!("candle mnist training...");
        let num_classes = 10;

        let device = &candle_core::Device::cuda_if_available(gpu_id)?;
        println!("device: {:?}", device);
        let dtype = DType::F32;

        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, dtype, device);
        let model = ConvNet::new(vs.clone(), num_classes).unwrap();
        let mut optimizer = candle_nn::SGD::new(varmap.all_vars(), learning_rate)?;

        let dataset = candle_datasets::vision::mnist::load()?;
        let train_images = &dataset.train_images.to_device(device)?;
        let train_labels = &dataset.train_labels.to_device(device)?;
        let test_images = &dataset.test_images.to_device(device)?;
        let test_labels = &dataset
            .test_labels
            .to_dtype(DType::U32)?
            .to_device(device)?;

        let n_batches = train_images.dim(0)? / batch_size;
        let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
        let start = Instant::now();
        for i in 0..num_epochs {
            let start = Instant::now();
            batch_idxs.shuffle(&mut thread_rng());
            let mut sum_loss = 0f32;
            for batch_idx in &batch_idxs {
                let train_images = &train_images.narrow(0, batch_idx * batch_size, batch_size)?;
                let train_labels = &train_labels.narrow(0, batch_idx * batch_size, batch_size)?;
                let logits = model.forward(train_images, true)?;
                let loss = loss_fn(&logits, train_labels)?;
                optimizer.backward_step(&loss)?;
                let loss = loss.to_vec0::<f32>()?;
                sum_loss += loss;
            }
            let avg_loss = sum_loss / n_batches as f32;
            let test_logits = model.forward(test_images, false)?;
            let sum_ok = test_logits
                .argmax(D::Minus1)?
                .eq(test_labels)?
                .to_dtype(DType::F32)?
                .sum_all()?
                .to_scalar::<f32>()?;
            let test_accuracy = sum_ok / test_labels.dims1()? as f32;
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
        Ok(())
    }
}

#[derive(ValueEnum, Clone)]
enum Which {
    Rai,
    Candle,
}

#[derive(Parser)]
struct Args {
    #[clap(value_enum, default_value_t = Which::Rai)]
    which: Which,

    #[arg(long, default_value_t = 10)]
    epochs: usize,

    #[arg(long, default_value_t = 0.05)]
    learning_rate: f64,

    #[arg(long, default_value_t = 0)]
    gpu_id: usize,
}

fn main() {
    let args = Args::parse();
    let batch_size = 64;

    match args.which {
        Which::Rai => {
            rai_mnist::training(args.epochs, args.learning_rate, batch_size, args.gpu_id);
        }
        Which::Candle => {
            candle_mnist::training(args.epochs, args.learning_rate, batch_size, args.gpu_id)
                .unwrap();
        }
    }
}
