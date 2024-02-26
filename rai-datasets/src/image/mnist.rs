use std::fs::File;

use crate::{Error, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use rai::{AsDevice, Tensor, F32};

#[derive(Debug, Clone)]
pub struct Dataset {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
    pub labels: usize,
}

fn load_parquet(
    parquet: SerializedFileReader<std::fs::File>,
    device: impl AsDevice,
) -> Result<(Tensor, Tensor)> {
    let device = device.device();
    let samples = parquet.metadata().file_metadata().num_rows() as usize;
    let mut buffer_images: Vec<u8> = Vec::with_capacity(samples * 784);
    let mut buffer_labels: Vec<u8> = Vec::with_capacity(samples);
    for row in parquet.into_iter().flatten() {
        for (_name, field) in row.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                buffer_labels.push(*label as u8);
            }
        }
    }
    let images = Tensor::from_array(buffer_images, [samples, 784], device).to_dtype(F32) / 255.;
    let labels = Tensor::from_array(buffer_labels, [samples], device);
    Ok((images, labels))
}

pub fn load(device: impl AsDevice) -> Result<Dataset> {
    let device = device.device();
    let api = Api::new().map_err(Error::wrap)?;
    let dataset_id = "mnist".to_string();
    let repo = Repo::with_revision(
        dataset_id,
        RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    );
    let repo = api.repo(repo);
    let test_parquet_filename = repo.get("mnist/test/0000.parquet")?;
    let train_parquet_filename = repo.get("mnist/train/0000.parquet")?;
    let test_parquet = SerializedFileReader::new(File::open(test_parquet_filename)?)?;
    let train_parquet = SerializedFileReader::new(File::open(train_parquet_filename)?)?;
    let (test_images, test_labels) = load_parquet(test_parquet, device)?;
    let (train_images, train_labels) = load_parquet(train_parquet, device)?;
    Ok(Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    })
}
