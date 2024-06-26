use hf_hub::{api::sync::Api, Repo, RepoType};
use rai::{device, ext, nn::Module, AsDevice, Device, Result, Tensor, Type, F32};
use rai_models::llm::{
    phi3::{Config, Model},
    utils::TokenOutputStream,
};
use std::{io::Write, time::Instant};
use tokenizers::Tokenizer;

fn load_model(dtype: impl Type, device: impl AsDevice) -> Result<(Tokenizer, Model)> {
    let start = Instant::now();
    let device = device.device();
    let model_id = "microsoft/Phi-3-mini-4k-instruct".to_string();
    let revision = "main".to_string();
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    // tokenizer
    let tokenizer_filename = repo.get("tokenizer.json").unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
    // model
    let config_filename = repo.get("config.json").unwrap();
    let config = std::fs::read_to_string(config_filename).unwrap();
    let cfg: Config = serde_json::from_str(&config).unwrap();
    let model_filenames = ext::hf::load_safetensors(&repo, "model.safetensors.index.json")?;
    let model = Model::new(&cfg, dtype, device);
    model.update_by_safetensors(&model_filenames, device)?;
    let elapsed = start.elapsed();
    println!("model loaded in : {:?}", elapsed);
    Ok((tokenizer, model))
}

fn sample_argmax(logits: &[f32]) -> u32 {
    let next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, u), (_, v)| u.total_cmp(v))
        .map(|(i, _)| i as u32)
        .unwrap();
    next_token
}

pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let context: std::collections::HashSet<_> = context.iter().collect();
    for (token_id, logit) in logits.iter_mut().enumerate() {
        if context.contains(&(token_id as u32)) {
            if *logit >= 0.0 {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
}

fn main() -> Result<()> {
    let device: Box<dyn Device> = device::cuda_if_available(0);
    let device = device.as_ref();
    let dtype = F32;
    let (tokenizer, model) = load_model(dtype, device)?;

    let prompt = "A skier slides down a frictionless slope of height 40m and length 80m. What's the skier speed at the bottom?";
    println!("{prompt}");
    std::io::stdout().flush().unwrap();
    let eos_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
        Some(token) => *token,
        None => panic!("cannot find the endoftext token"),
    };

    let mut output_stream = TokenOutputStream::new(tokenizer);
    output_stream.prompt(prompt);

    // let tokens = tokenizer.encode(prompt, true).unwrap();
    // let mut tokens = tokens.get_ids().to_vec();
    let mut generated_tokens = 0usize;

    let start_gen = std::time::Instant::now();
    let sample_len: usize = 5000;
    let repeat_penalty: f32 = 1.10;
    let repeat_last_n: usize = 64;

    let mut pos = 0;
    for index in 0..sample_len {
        let context_size = if index > 0 { 1 } else { output_stream.len() };
        let ctx = &output_stream.tokens()[output_stream.len().saturating_sub(context_size)..];
        let input = Tensor::from_array(ctx, [ctx.len()], device).unsqueeze(0);
        let logits = model.fwd(&input, pos).narrow(1, 0, 1).squeeze(1);
        let logits = logits.squeeze(0);
        let mut logits = logits.as_vec(dtype)?;
        if repeat_penalty >= 1. {
            let start_at = output_stream.len().saturating_sub(repeat_last_n);
            apply_repeat_penalty(
                &mut logits,
                repeat_penalty,
                &output_stream.tokens()[start_at..],
            )
        };
        let next_token = sample_argmax(&logits);
        generated_tokens += 1;
        if next_token == eos_token {
            break;
        }

        if let Some(t) = output_stream.next_token(next_token) {
            print!("{t}");
            std::io::stdout().flush().unwrap();
        }
        pos += context_size;
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{generated_tokens} tokens generated ({:.2} token/s)",
        generated_tokens as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
