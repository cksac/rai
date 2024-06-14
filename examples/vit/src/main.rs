use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rai::{device, nn::Module, AsDevice, Device, Result, Tensor, Type, F32};
use rai_models::cv::vit::{ImageClassificationConfig, ImageClassificationModel};
use std::time::Instant;

pub fn load_image(url: impl AsRef<str>, dtype: impl Type, device: impl AsDevice) -> Tensor {
    let device = device.device();
    let img_bytes = reqwest::blocking::get(url.as_ref())
        .expect("request failed")
        .bytes()
        .unwrap();
    let img = image::load_from_memory(&img_bytes).unwrap().resize_to_fill(
        224,
        224,
        image::imageops::FilterType::Triangle,
    );

    // todo: add ViT ImageProcessor
    // https://huggingface.co/google/vit-base-patch16-224/blob/main/preprocessor_config.json
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_array(data, [224, 224, 3], device)
        .permute([2, 0, 1])
        .to_dtype(dtype);
    let mean = Tensor::from_array(vec![0.485f32, 0.456, 0.406], [3, 1, 1], device);
    let std = Tensor::from_array(vec![0.229f32, 0.224, 0.225], [3, 1, 1], device);
    (data / 255.0 - mean) / std
}

fn load_model(
    dtype: impl Type,
    device: impl AsDevice,
) -> Result<(ImageClassificationConfig, ImageClassificationModel)> {
    let start = Instant::now();
    let device = device.device();
    let model_id = "google/vit-base-patch16-224".to_string();
    let revision = "main".to_string();
    let api = Api::new().unwrap();
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let config_filename = repo.get("config.json").unwrap();
    let config = std::fs::read_to_string(config_filename).unwrap();
    let cfg: ImageClassificationConfig = serde_json::from_str(&config).unwrap();
    let model_filenames = vec![repo.get("model.safetensors").unwrap()];
    let model = ImageClassificationModel::new(&cfg, dtype, device);
    model.update_by_safetensors(&model_filenames, device)?;
    let elapsed = start.elapsed();
    println!("model loaded in : {:?}", elapsed);
    Ok((cfg, model))
}

#[derive(Parser)]
struct Args {
    #[arg(
        long,
        default_value = "http://images.cocodataset.org/val2017/000000039769.jpg"
    )]
    image_url: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device: Box<dyn Device> = device::cuda_if_available(0);
    let device = device.as_ref();
    let dtype = F32;
    let (cfg, model) = load_model(dtype, device)?;
    let image = load_image(args.image_url, dtype, device);
    let logits = model.forward(&image.unsqueeze(0));
    let prs = logits.softmax(-1).narrow(0, 0, 1).squeeze(0).as_vec(F32)?;
    let mut prs = prs.iter().enumerate().collect::<Vec<_>>();
    prs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for &(category_idx, pr) in prs.iter().take(5) {
        println!(
            "{:24}: {:.2}%",
            cfg.id2label.get(&category_idx).unwrap(),
            100. * pr
        );
    }
    Ok(())
}
