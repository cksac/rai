mod linear;
pub use linear::Linear;

mod activations;
pub use activations::*;

mod embedding;
pub use embedding::*;

mod layer_norm;
pub use layer_norm::*;

#[macro_export]
macro_rules! gather_params {
    ($P:expr, $M:ident) => {
        $M.insert($P.id(), $P.clone());
    };

    (?$P:expr, $M:ident) => {
        if let Some(p) = &$P {
            $M.insert(p.id(), p.clone());
        }
    };

    (@$L:expr, $M:ident) => {
        $L.gather_params($M);
    };

    ([]$L:expr, $M:ident) => {
        for l in &$L {
            l.gather_params($M);
        }
    };
}

#[macro_export]
macro_rules! update_params {
    ($P:expr, $M:ident) => {
        if let Some(p) = $M.remove(&$P.id()) {
            $P.replace_data(p);
        }
    };

    (?$P:expr, $M:ident) => {
        if let Some(p) = &$P {
            if let Some(np) = $M.remove(&p.id()) {
                p.replace_data(np);
            }
        }
    };

    (@$L:expr, $M:ident) => {
        $L.update_params($M);
    };

    ([]$L:expr, $M:ident) => {
        for l in &$L {
            l.update_params($M);
        }
    };
}
