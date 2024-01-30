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
        // todo: return result
        pub fn load_safetensors(
            repo: &hf_hub::api::sync::ApiRepo,
            json_file: &str,
        ) -> Vec<std::path::PathBuf> {
            let json_file = repo.get(json_file).unwrap();
            let json_file = std::fs::File::open(json_file).unwrap();
            let json: serde_json::Value = serde_json::from_reader(&json_file).unwrap();
            let weight_map = match json.get("weight_map") {
                None => panic!("no weight map in {json_file:?}"),
                Some(serde_json::Value::Object(map)) => map,
                Some(_) => panic!("weight map in {json_file:?} is not a map"),
            };
            let mut safetensors_files = std::collections::HashSet::new();
            for value in weight_map.values() {
                if let Some(file) = value.as_str() {
                    safetensors_files.insert(file.to_string());
                }
            }
            let safetensors_files = safetensors_files
                .iter()
                .map(|v| repo.get(v).unwrap())
                .collect::<Vec<_>>();

            safetensors_files
        }
    }
}
