pub use rai_core::*;

pub mod nn {
    pub use rai_core::nn::*;
    pub use rai_nn::*;
}

pub mod opt {
    pub use rai_opt::*;
}

pub use rai_derive::Module;

pub mod ext {
    pub mod hf {
        use hf_hub::api::sync::ApiError;
        pub fn load_safetensors(
            repo: &hf_hub::api::sync::ApiRepo,
            json_file_path: &str,
        ) -> rai_core::Result<Vec<std::path::PathBuf>> {
            let json_file = repo.get(json_file_path)?;
            let json_file = std::fs::File::open(json_file)?;
            let json: serde_json::Value = serde_json::from_reader(&json_file)?;
            let weight_map = match json.get("weight_map") {
                None => return Err(rai_core::Error::MissingWeightMap),
                Some(serde_json::Value::Object(map)) => map,
                Some(_) => {
                    return Err(rai_core::Error::InvalidWeightMap(json_file_path.to_owned()))
                }
            };
            let mut safetensors_files = std::collections::HashSet::new();
            for value in weight_map.values() {
                if let Some(file) = value.as_str() {
                    safetensors_files.insert(file.to_string());
                }
            }
            safetensors_files
                .iter()
                .map(|v| repo.get(v))
                .collect::<Result<Vec<_>, ApiError>>()
                .map_err(Into::into)
        }
    }
}
