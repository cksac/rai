mod linear;
use std::{borrow::Cow, collections::HashMap};

pub use linear::Linear;

mod activations;
pub use activations::*;

mod embedding;
pub use embedding::*;

mod layer_norm;
pub use layer_norm::*;

mod rms_norm;
use rai_core::{eval, Shape, Tensor};
pub use rms_norm::*;

#[macro_export]
macro_rules! gather_params {
    ($M:ident, $P:expr) => {
        $M.insert($P.id(), $P.clone());
    };

    ($M:ident, ?$P:expr) => {
        if let Some(p) = &$P {
            $M.insert(p.id(), p.clone());
        }
    };

    ($M:ident, @$L:expr) => {
        $L.gather_params($M);
    };

    ($M:ident, []$L:expr) => {
        for l in &$L {
            l.gather_params($M);
        }
    };
}

#[macro_export]
macro_rules! update_params {
    ($M:ident, $P:expr) => {
        if let Some(p) = $M.remove(&$P.id()) {
            $P.replace_data(p);
        }
    };

    ($M:ident, ?$P:expr) => {
        if let Some(p) = &$P {
            if let Some(np) = $M.remove(&p.id()) {
                p.replace_data(np);
            }
        }
    };

    ($M:ident, @$L:expr) => {
        $L.update_params($M);
    };

    ($M:ident, []$L:expr) => {
        for l in &$L {
            l.update_params($M);
        }
    };
}

pub trait NamedParameter {
    fn gather_to(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str);
    fn update_by(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str);
}

impl NamedParameter for Tensor {
    fn gather_to(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        let name = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name)
        };
        params.insert(name, self.clone());
    }

    fn update_by(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        let name: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        if let Some(t) = params.remove(name.as_ref()) {
            if self.shape() != t.shape() {
                panic!(
                    "parameter {} shape {:?} not align with data shape {:?}",
                    name,
                    self.shape(),
                    t.shape()
                );
            }

            if self.device() != t.device() {
                panic!(
                    "parameter {} device {:?} not align with data device {:?}",
                    name,
                    self.device(),
                    t.device()
                );
            }

            // todo: check if can promote type to self
            let t = t.as_type(self);
            eval((&t, true));
            self.replace_data(t);
        } else {
            panic!("parameter {} not found", name);
        }
    }
}

impl NamedParameter for Option<Tensor> {
    fn gather_to(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        if let Some(t) = self {
            t.gather_to(params, prefix, name);
        }
    }

    fn update_by(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        if let Some(t) = self {
            t.update_by(params, prefix, name)
        }
    }
}

pub trait NamePath {
    fn push<'b>(&self, path: &'b str) -> Cow<'b, str>;
}

impl<'a> NamePath for &'a str {
    fn push<'b>(&self, path: &'b str) -> Cow<'b, str> {
        if self.is_empty() {
            path.into()
        } else {
            format!("{}.{}", self, path).into()
        }
    }
}

impl<'a> NamePath for Cow<'a, str> {
    fn push<'b>(&self, path: &'b str) -> Cow<'b, str> {
        if self.is_empty() {
            path.into()
        } else {
            format!("{}.{}", self, path).into()
        }
    }
}
