mod linear;
pub use linear::Linear;

mod activations;
pub use activations::*;

mod embedding;
pub use embedding::*;

mod layer_norm;
pub use layer_norm::*;

mod rms_norm;
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

#[macro_export]
macro_rules! gather_named_params {
    ($M:ident, $NP:expr, $N:expr, $P:expr) => {
        $M.insert(format!("{}.{}", $NP, $N), $P.clone());
    };

    ($M:ident, $NP:expr, $N:expr, ?$P:expr) => {
        if let Some(p) = &$P {
            $M.insert(format!("{}.{}", $NP, $N), p.clone());
        }
    };

    ($M:ident, $NP:expr, $N:expr, @$L:expr) => {
        let p = format!("{}.{}", $NP, $N);
        $L.gather_named_params(&p, $M);
    };

    ($M:ident, $NP:expr, $N:expr, []$L:expr) => {
        for (i, l) in $L.iter().enumerate() {
            let p = format!("{}.{}.{}", $NP, $N, i);
            l.gather_named_params(&p, $M);
        }
    };
}
