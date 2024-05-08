use rai::{
    device, eval,
    nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module},
    opt::{
        losses,
        optimizers::{Optimizer, SDG},
    },
    value_and_grad, AsDevice, Device, FloatElemType, Module, Shape, Tensor, Type, F32, U32,
};
use rai_datasets::image::mnist;
use rand::{seq::SliceRandom, thread_rng};
use std::time::Instant;

#[derive(Clone, Debug, Module)]
#[module(trainable = false)]
struct TimeEmbedding {
    half_emb: Tensor,
}

impl TimeEmbedding {
    pub fn new(emb_size: usize, dtype: impl Type, device: impl AsDevice) -> TimeEmbedding {
        let size = (emb_size / 2) as f32;
        let half_emb = (Tensor::arange(size, device) * (-1.0 * 10000f32.ln() / (size - 1.0)))
            .exp()
            .to_dtype(dtype);
        TimeEmbedding { half_emb }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let l = x.shape_at(0);
        let x = x.reshape([l, 1]);
        let emb = self
            .half_emb
            .unsqueeze(0)
            .broadcast_to([l, self.half_emb.shape_at(0)]);
        let t = emb * x.to_dtype(&self.half_emb);
        Tensor::cat(&[t.sin(), t.cos()], -1)
    }
}

#[derive(Clone, Debug, Module)]
#[module(input = (Tensor, Tensor))]
struct DiTBlock {
    gamma1: Linear,
    beta1: Linear,
    alpha1: Linear,
    gamma2: Linear,
    beta2: Linear,
    alpha2: Linear,
    ln1: LayerNorm,
    ln2: LayerNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    lv: Linear,
    ff_ln1: Linear,
    ff_ln2: Linear,
    #[param(skip)]
    emb_size: usize,
    #[param(skip)]
    nhead: usize,
}

impl DiTBlock {
    pub fn new(
        emb_size: usize,
        nhead: usize,
        eps: f64,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> DiTBlock {
        let device = device.device();
        // conditioning
        let gamma1 = Linear::new(emb_size, emb_size, true, dtype, device);
        let beta1 = Linear::new(emb_size, emb_size, true, dtype, device);
        let alpha1 = Linear::new(emb_size, emb_size, true, dtype, device);
        let gamma2 = Linear::new(emb_size, emb_size, true, dtype, device);
        let beta2 = Linear::new(emb_size, emb_size, true, dtype, device);
        let alpha2 = Linear::new(emb_size, emb_size, true, dtype, device);

        // layer norm
        let ln1 = LayerNorm::new(emb_size, eps, true, dtype, device);
        let ln2 = LayerNorm::new(emb_size, eps, true, dtype, device);

        // multi-head self-attention
        let wq = Linear::new(emb_size, emb_size * nhead, true, dtype, device);
        let wk = Linear::new(emb_size, emb_size * nhead, true, dtype, device);
        let wv = Linear::new(emb_size, emb_size * nhead, true, dtype, device);
        let lv = Linear::new(emb_size * nhead, emb_size, true, dtype, device);

        // feed forward
        let ff_ln1 = Linear::new(emb_size, emb_size * 4, true, dtype, device);
        let ff_ln2 = Linear::new(emb_size * 4, emb_size, true, dtype, device);

        DiTBlock {
            gamma1,
            beta1,
            alpha1,
            gamma2,
            beta2,
            alpha2,
            ln1,
            ln2,
            wq,
            wk,
            wv,
            lv,
            ff_ln1,
            ff_ln2,
            emb_size,
            nhead,
        }
    }

    // x:(batch,seq_len,emb_size), cond:(batch,emb_size)
    pub fn fwd(&self, x: &Tensor, cond: &Tensor) -> Tensor {
        let g1v = self.gamma1.forward(cond);
        let b1v = self.beta1.forward(cond);
        let a1v = self.alpha1.forward(cond);
        let g2v = self.gamma2.forward(cond);
        let b2v = self.beta2.forward(cond);
        let a2v = self.alpha2.forward(cond);

        let mut y = self.ln1.forward(x);
        y = y * (1 + g1v.unsqueeze(1)) + b1v.unsqueeze(1);
        let mut q = self.wq.forward(&y); // (batch,seq_len,nhead*emb_size)
        let mut k = self.wk.forward(&y); // (batch,seq_len,nhead*emb_size)
        let mut v = self.wv.forward(&y); // (batch,seq_len,nhead*emb_size)
        q = q
            .reshape([q.shape_at(0), q.shape_at(1), self.nhead, self.emb_size])
            .permute([0, 2, 1, 3]); // (batch,nhead,seq_len,emb_size)
        k = k
            .reshape([k.shape_at(0), k.shape_at(1), self.nhead, self.emb_size])
            .permute([0, 2, 3, 1]); // (batch,nhead,seq_len,emb_size)
        v = v
            .reshape([v.shape_at(0), v.shape_at(1), self.nhead, self.emb_size])
            .permute([0, 2, 1, 3]); // (batch,nhead,seq_len,emb_size)

        let attn = (q.matmul(k) / (q.shape_at(2) as f32).sqrt()).softmax(-1); // (batch,nhead,seq_len,seq_len)
        y = attn.matmul(v); // (batch,nhead,seq_len,emb_size)
        y = y.permute([0, 2, 1, 3]).flatten(2..); // (batch,seq_len,nhead*emb_size)
        y = self.lv.forward(&y); // (batch,seq_len,emb_size)
        y = y * a1v.unsqueeze(1);
        y = x + y;
        let mut z = self.ln2.forward(&y);
        z = z * (1 + g2v.unsqueeze(1)) + b2v.unsqueeze(1);
        z = z.apply(&self.ff_ln1).relu().apply(&self.ff_ln2);
        z = z * a2v.unsqueeze(1);
        y + z
    }
}

#[derive(Clone, Debug, Module)]
#[module(input = (Tensor, Tensor, Tensor))]
struct DiT {
    conv: Conv2d,
    patch_emb: Linear,
    patch_pos_emb: Tensor,
    time_emb: TimeEmbedding,
    time_emb_ln1: Linear,
    time_emb_ln2: Linear,
    label_emb: Embedding,
    dit_blocks: Vec<DiTBlock>,
    ln: LayerNorm,
    linear: Linear,
    #[param(skip)]
    patch_count: usize,
    #[param(skip)]
    patch_size: usize,
    #[param(skip)]
    channel: usize,
}

impl DiT {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        channel: usize,
        emb_size: usize,
        label_num: usize,
        dit_num: usize,
        head: usize,
        eps: f64,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> DiT {
        let device = device.device();
        let patch_count = img_size / patch_size;
        let conv_confg = Conv2dConfig {
            stride: [patch_size, patch_size],
            ..Default::default()
        };
        let conv = Conv2d::new(
            channel,
            channel * patch_size * patch_size,
            patch_size,
            conv_confg,
            true,
            dtype,
            device,
        );
        let patch_emb = Linear::new(
            channel * patch_size * patch_size,
            emb_size,
            true,
            dtype,
            device,
        );
        let patch_pos_emb = Tensor::rand([1, patch_count * patch_count, emb_size], dtype, device);
        let time_emb = TimeEmbedding::new(emb_size, dtype, device);
        let time_emb_ln1 = Linear::new(emb_size, emb_size, true, dtype, device);
        let time_emb_ln2 = Linear::new(emb_size, emb_size, true, dtype, device);
        let label_emb = Embedding::new(label_num, emb_size, dtype, device);
        let mut dit_blocks = Vec::with_capacity(dit_num);
        for _ in 0..dit_num {
            let b = DiTBlock::new(emb_size, head, eps, dtype, device);
            dit_blocks.push(b);
        }
        let ln = LayerNorm::new(emb_size, eps, true, dtype, device);
        let linear = Linear::new(
            emb_size,
            channel * patch_size * patch_size,
            true,
            dtype,
            device,
        );
        DiT {
            conv,
            patch_emb,
            patch_pos_emb,
            time_emb,
            time_emb_ln1,
            time_emb_ln2,
            label_emb,
            dit_blocks,
            ln,
            linear,
            patch_count,
            patch_size,
            channel,
        }
    }

    // x:(batch,channel,height,width)   t:(batch,)  y:(batch,)
    pub fn fwd(&self, x: &Tensor, t: &Tensor, y: &Tensor) -> Tensor {
        // label emb, (batch,emb_size)
        let y_emb = self.label_emb.forward(y);
        // time emb, (batch,emb_size)
        let t_emb = self
            .time_emb
            .fwd(t)
            .apply(&self.time_emb_ln1)
            .relu()
            .apply(&self.time_emb_ln2);

        let cond = y_emb + t_emb;

        // patch emb
        let mut x = self.conv.forward(x); // (batch,new_channel,patch_count,patch_count)
        x = x.permute([0, 2, 3, 1]); // (batch,patch_count,patch_count,new_channel)
        x = x.reshape([
            x.shape_at(0),
            self.patch_count * self.patch_count,
            x.shape_at(3),
        ]); // (batch,patch_count**2,new_channel)
        x = x.apply(&self.patch_emb) + &self.patch_pos_emb; // (batch,patch_count**2,emb_size)

        for b in &self.dit_blocks {
            x = b.fwd(&x, &cond);
        }

        x = x.apply(&self.ln); // (batch,patch_count**2,emb_size)
        x = x.apply(&self.linear); //  (batch,patch_count**2,channel*patch_size*patch_size)

        x = x.reshape([
            x.shape_at(0),
            self.patch_count,
            self.patch_count,
            self.channel,
            self.patch_size,
            self.patch_size,
        ]); //  (batch,patch_count,patch_count,channel,patch_size,patch_size)

        //x = x.permute([0, 3, 1, 2, 4, 5]); // (batch,channel,patch_count(H),patch_count(W),patch_size(H),patch_size(W))
        //x = x.permute([0, 1, 2, 4, 3, 5]); // (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x = x.permute([0, 3, 1, 4, 2, 5]); // (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x = x.reshape([
            x.shape_at(0),
            self.channel,
            self.patch_count * self.patch_size,
            self.patch_count * self.patch_size,
        ]); // (batch,channel,img_size,img_size)
        x
    }
}

#[test]
fn test_time_emb() {
    let dtype = F32;
    let device = rai::Cpu;
    let time_emb = TimeEmbedding::new(16, dtype, device);
    let x = Tensor::rand_with(0f32, 1000.0, [2], device).to_dtype(U32);
    let y = time_emb.fwd(&x);
    println!("{}", y);
}

#[test]
fn test_dit_block() {
    let dtype = F32;
    let device = rai::Cpu;
    let dit_block = DiTBlock::new(16, 4, 1e-4, dtype, device);
    let x = Tensor::rand([5, 49, 16], dtype, device);
    let cond = Tensor::rand([5, 16], dtype, device);
    let y = dit_block.fwd(&x, &cond);
    println!("{}", y);
}

#[test]
fn test_dit_model() {
    let dtype = F32;
    let device = rai::Cpu;
    let dit = DiT::new(28, 4, 1, 64, 10, 1, 4, 1e4, dtype, device);
    let x = Tensor::rand([5, 1, 28, 28], dtype, device);
    let t = Tensor::rand_with(0f32, 1000.0, [5], device).to_dtype(U32);
    let y = Tensor::rand_with(0f32, 10.0, [5], device).to_dtype(U32);
    let output = dit.fwd(&x, &t, &y);
    println!("{}", output);
}

struct DDIMScheduler {
    alphas_cumprod: Tensor,
}

impl DDIMScheduler {
    pub fn new(_dtype: impl Type, device: impl AsDevice) -> DDIMScheduler {
        // betas=torch.linspace(0.0001,0.02,T) # (T,)
        // alphas=1-betas  # (T,)
        // alphas_cumprod=torch.cumprod(alphas,dim=-1) # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
        // alphas_cumprod_prev=torch.cat((torch.tensor([1.0]),alphas_cumprod[:-1]),dim=-1) # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
        // variance=(1-alphas)*(1-alphas_cumprod_prev)/(1-alphas_cumprod)  # denoise用的方差   (T,)

        let betas = f32::linspace(0.0001, 0.02, 1000);
        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        for &beta in betas.iter() {
            let alpha = 1.0 - beta;
            alphas_cumprod.push(alpha * *alphas_cumprod.last().unwrap_or(&1f32))
        }
        let alphas_cumprod = Tensor::from_array(alphas_cumprod, [1000], device);
        Self { alphas_cumprod }
    }

    pub fn add_noise(&self, x: &Tensor, t: &Tensor) -> (Tensor, Tensor) {
        // noise=torch.randn_like(x)   # 为每张图片生成第t步的高斯噪音   (batch,channel,height,width)
        // batch_alphas_cumprod=alphas_cumprod[t].view(x.size(0),1,1,1)
        // x=torch.sqrt(batch_alphas_cumprod)*x+torch.sqrt(1-batch_alphas_cumprod)*noise # 基于公式直接生成第t步加噪后图片
        // return x,noise

        let noise = Tensor::randn_like(x);
        let batch_alphas_cumprod = self.alphas_cumprod.index_select(0, t).to_dtype(x);
        let batch_alphas_cumprod = batch_alphas_cumprod.reshape([x.shape_at(0), 1, 1, 1]);
        let x = batch_alphas_cumprod.sqrt() * x + (1.0f32 - batch_alphas_cumprod).sqrt() * &noise;
        (x, noise)
    }
}

fn loss_fn(model: &DiT, x: &Tensor, t: &Tensor, y: &Tensor, noise: &Tensor) -> Tensor {
    let pred_noise = model.fwd(x, t, y);
    losses::l1_loss(&pred_noise, noise).mean(..)
}

fn train(
    model: &DiT,
    scheduler: &DDIMScheduler,
    num_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    device: impl AsDevice,
) {
    let device = device.device();
    let params = model.params();
    println!("params: {:?}", params.len());
    let mut optimizer = SDG::new(params, learning_rate);
    let dataset = mnist::load(device).expect("mnist dataset");
    let train_images = dataset.train_images;
    let train_images = train_images.reshape([train_images.shape_at(0), 1, 28, 28]);
    let train_labels = &dataset.train_labels;
    let vg_fn = value_and_grad(loss_fn);
    let n_batches = train_images.shape_at(0) / batch_size;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();
    let mut iter_cnt = 0;
    let start = Instant::now();
    for i in 0..num_epochs {
        let start = Instant::now();
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in &batch_idxs {
            let images = &train_images.narrow(0, batch_idx * batch_size, batch_size);
            let labels = &train_labels.narrow(0, batch_idx * batch_size, batch_size);
            let x = images * 2.0 - 1.0; // convert image range from [0,1] to [-1,1], align with noise range
            let t = Tensor::rand_with(0.0f32, 1000.0, [batch_size], device).to_dtype(U32);
            let (xs, noise) = scheduler.add_noise(&x, &t);
            let (loss, (grads, ..)) = vg_fn((model, &xs, &t, labels, &noise));
            let mut params = optimizer.step(&grads);
            eval(&params);
            model.update_params(&mut params);
            iter_cnt += 1;
            if iter_cnt % 1000 == 0 && iter_cnt > 0 {
                println!(
                    "epoch: {i:04}, iter: {iter_cnt}, loss: {:10.5}",
                    loss.as_scalar(F32)
                );
            }
        }
        let elapsed = start.elapsed();
        println!("epoch: {i:04}, time: {elapsed:?}");
    }
    let elapsed = start.elapsed();
    let avg_elapsed = elapsed.as_secs_f64() / num_epochs as f64;
    println!("elapsed: {:?}, avg: {:.2} sec/epoch", elapsed, avg_elapsed);
    model.to_safetensors("mnist-dit.safetensors");
}

fn main() {
    let num_epochs = 1;
    let learning_rate = 0.05;
    let batch_size = 300;

    let dtype = F32;
    let device: Box<dyn Device> = device::cuda_if_available(0);
    let device = device.as_ref();
    println!("device: {:?}", device);

    let model = DiT::new(28, 4, 1, 64, 10, 3, 4, 1e4, dtype, device);
    let scheduler = DDIMScheduler::new(dtype, device);

    train(
        &model,
        &scheduler,
        num_epochs,
        batch_size,
        learning_rate,
        device,
    );
}
