use std::collections::{BTreeMap, HashMap};

use rai_core::{Shape, Tensor};

use super::Optimizer;

pub struct SDG {
    lr: f32,
    momentum: Option<f32>,
    weight_decay: Option<f32>,
    dampening: Option<f32>,
    nesterov: Option<f32>,
    state: HashMap<usize, Tensor>,
}

impl SDG {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: None,
            weight_decay: None,
            dampening: None,
            nesterov: None,
            state: HashMap::new(),
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = Some(momentum);
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = Some(weight_decay);
        self
    }

    pub fn with_dampening(mut self, dampening: f32) -> Self {
        self.dampening = Some(dampening);
        self
    }

    pub fn with_nesterov(mut self, nesterov: f32) -> Self {
        self.nesterov = Some(nesterov);
        self
    }
}

impl Optimizer for SDG {
    fn step(
        &mut self,
        params: Vec<Tensor>,
        grads: &BTreeMap<usize, Tensor>,
    ) -> BTreeMap<usize, Tensor> {
        let mut new_params = BTreeMap::new();
        for p in params.iter() {
            let id = p.id();
            let mut g: Tensor = grads.get(&id).cloned().unwrap();
            // TODO: check why grad of Linear.bias is [BATCH_SIZE, LABEL_DIM] instead of [LABEL_DIM] where input is batched, bug in vjp?
            if !g.shape_eq(p) {
                g = g.mean(..p.ndim());
            }

            let new_p = match self.momentum {
                Some(momentum) => {
                    let mut v: Tensor =
                        self.state.get(&id).cloned().unwrap_or(g.zeros_like()) * momentum;
                    if let Some(weight_decay) = self.weight_decay {
                        g = &g + p * weight_decay;
                    }

                    match self.dampening {
                        Some(dampening) => {
                            v = v + &g * (1.0 - dampening);
                        }
                        None => {
                            v = v + &g;
                        }
                    }

                    let new_p = match self.nesterov {
                        Some(nesterov) => p - &v * self.lr * nesterov,
                        None => p - &v * self.lr,
                    };

                    self.state.insert(id, v);
                    new_p
                }
                None => p - self.lr * g,
            };

            new_params.insert(id, new_p);
        }
        new_params
    }
}
