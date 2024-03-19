use rai_core::{AsDevice, Shape, Tensor, Type, F64};

// ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py

pub trait Init {
    fn new_tensor(&self, shape: impl Shape, dtype: impl Type, device: impl AsDevice) -> Tensor;
}

pub const DEFAULT_KAIMING_UNIFORM: Kaiming = Kaiming {
    dist: NormalOrUniform::Uniform,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

pub const DEFAULT_KAIMING_NORMAL: Kaiming = Kaiming {
    dist: NormalOrUniform::Normal,
    fan: FanInOut::FanIn,
    non_linearity: NonLinearity::ReLU,
};

#[derive(Debug, Copy, Clone)]
pub enum FanInOut {
    FanIn,
    FanOut,
}

impl FanInOut {
    pub fn for_shape(&self, shape: impl Shape) -> usize {
        let dims = shape.shape();
        let receptive_field_size: usize = dims.iter().skip(2).product();
        match &self {
            FanInOut::FanIn => {
                if dims.len() < 2 {
                    1
                } else {
                    dims[1] * receptive_field_size
                }
            }
            FanInOut::FanOut => {
                if dims.is_empty() {
                    1
                } else {
                    dims[0] * receptive_field_size
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NormalOrUniform {
    Normal,
    Uniform,
}

#[derive(Debug, Copy, Clone)]
pub enum NonLinearity {
    ReLU,
    Linear,
    Sigmoid,
    Tanh,
    SELU,
    ExplicitGain(f64),
}

impl NonLinearity {
    pub fn gain(&self) -> f64 {
        match *self {
            NonLinearity::ReLU => 2f64.sqrt(),
            NonLinearity::Tanh => 5. / 3.,
            NonLinearity::Linear | NonLinearity::Sigmoid => 1.,
            NonLinearity::SELU => 0.75,
            NonLinearity::ExplicitGain(g) => g,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Kaiming {
    dist: NormalOrUniform,
    fan: FanInOut,
    non_linearity: NonLinearity,
}

impl Init for Kaiming {
    fn new_tensor(&self, shape: impl Shape, dtype: impl Type, device: impl AsDevice) -> Tensor {
        let fan = self.fan.for_shape(&shape);
        let gain = self.non_linearity.gain();
        let std = gain / (fan as f64).sqrt();
        match self.dist {
            NormalOrUniform::Uniform => {
                let bound = 3f64.sqrt() * std;
                Tensor::rand_with(-bound, bound, shape, F64, device).to_dtype(dtype)
            }
            NormalOrUniform::Normal => {
                Tensor::randn_with(0.0, std, shape, F64, device).to_dtype(dtype)
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Uniform {
    from: f64,
    to: f64,
}

impl Uniform {
    pub fn new(from: f64, to: f64) -> Self {
        Self { from, to }
    }
}

impl Init for Uniform {
    fn new_tensor(&self, shape: impl Shape, dtype: impl Type, device: impl AsDevice) -> Tensor {
        Tensor::rand_with(self.from, self.to, shape, F64, device).to_dtype(dtype)
    }
}
