use crate::init::{self, Init, DEFAULT_KAIMING_NORMAL};
use rai_core::{AsDevice, Shape, Tensor, Type};
use rai_derive::Module;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    #[param(skip)]
    config: Conv1dConfig,
}

pub trait IntoConv1dConfig: Debug {
    fn into_conv1d_config(self) -> Conv1dConfig;
}

impl IntoConv1dConfig for Conv1dConfig {
    fn into_conv1d_config(self) -> Conv1dConfig {
        self
    }
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: impl IntoConv1dConfig,
        has_bias: bool,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let device = device.device();
        let config = config.into_conv1d_config();
        let weight = Tensor::rand(
            [out_channels, in_channels / config.groups, kernel_size],
            dtype,
            device,
        );
        let bias = if has_bias {
            Some(Tensor::rand([out_channels], dtype, device))
        } else {
            None
        };
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let x = x.conv1d(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        );
        match &self.bias {
            Some(bias) => {
                let bias = bias.reshape([1, bias.shape_at(0), 1, 1]);
                x + bias
            }
            None => x,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv2dConfig {
    pub padding: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    #[param(skip)]
    config: Conv2dConfig,
}

pub trait IntoConv2dConfig: Debug {
    fn into_conv2d_config(self) -> Conv2dConfig;
}

impl IntoConv2dConfig for Conv2dConfig {
    fn into_conv2d_config(self) -> Conv2dConfig {
        self
    }
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: impl IntoConv2dConfig,
        has_bias: bool,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let bound = 1. / (in_channels as f64).sqrt();
        let bias_init = match has_bias {
            true => Some(init::Uniform::new(-bound, bound)),
            false => None,
        };
        Self::new_with_init(
            in_channels,
            out_channels,
            kernel_size,
            config,
            dtype,
            device,
            DEFAULT_KAIMING_NORMAL,
            bias_init,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_init(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: impl IntoConv2dConfig,
        dtype: impl Type,
        device: impl AsDevice,
        weight_init: impl Init,
        bias_init: Option<impl Init>,
    ) -> Self {
        let device = device.device();
        let config = config.into_conv2d_config();
        let weight = weight_init.new_tensor(
            [
                out_channels,
                in_channels / config.groups,
                kernel_size,
                kernel_size,
            ],
            dtype,
            device,
        );
        let bias = bias_init.map(|init| init.new_tensor([out_channels], dtype, device));
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let x = x.conv2d(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        );
        match &self.bias {
            Some(bias) => {
                let bias = bias.reshape([1, bias.shape_at(0), 1, 1]);
                x + bias
            }
            None => x,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose1dConfig {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for ConvTranspose1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    #[param(skip)]
    config: ConvTranspose1dConfig,
}

pub trait IntoConvTranspose1dConfig: Debug {
    fn into_conv_transpose1d_config(self) -> ConvTranspose1dConfig;
}

impl IntoConvTranspose1dConfig for ConvTranspose1dConfig {
    fn into_conv_transpose1d_config(self) -> ConvTranspose1dConfig {
        self
    }
}

impl ConvTranspose1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: impl IntoConvTranspose1dConfig,
        has_bias: bool,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let device = device.device();
        let config = config.into_conv_transpose1d_config();
        let weight = Tensor::rand(
            [in_channels, out_channels / config.groups, kernel_size],
            dtype,
            device,
        );
        let bias = if has_bias {
            Some(Tensor::rand([out_channels], dtype, device))
        } else {
            None
        };
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let x = x.conv_transpose1d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        );
        match &self.bias {
            Some(bias) => {
                let bias = bias.reshape([1, bias.shape_at(0), 1, 1]);
                x + bias
            }
            None => x,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dConfig {
    pub padding: [usize; 2],
    pub output_padding: [usize; 2],
    pub stride: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

impl Default for ConvTranspose2dConfig {
    fn default() -> Self {
        Self {
            padding: [0, 0],
            output_padding: [0, 0],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

#[derive(Clone, Debug, Module)]
#[module(crate = rai_core)]
pub struct ConvTranspose2d {
    weight: Tensor,
    bias: Option<Tensor>,
    #[param(skip)]
    config: ConvTranspose2dConfig,
}

pub trait IntoConvTranspose2dConfig: Debug {
    fn into_conv_transpose2d_config(self) -> ConvTranspose2dConfig;
}

impl IntoConvTranspose2dConfig for ConvTranspose2dConfig {
    fn into_conv_transpose2d_config(self) -> ConvTranspose2dConfig {
        self
    }
}

impl ConvTranspose2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: impl IntoConvTranspose2dConfig,
        has_bias: bool,
        dtype: impl Type,
        device: impl AsDevice,
    ) -> Self {
        let device = device.device();
        let config = config.into_conv_transpose2d_config();
        let weight = Tensor::rand(
            [
                in_channels,
                out_channels / config.groups,
                kernel_size,
                kernel_size,
            ],
            dtype,
            device,
        );
        let bias = if has_bias {
            Some(Tensor::rand([out_channels], dtype, device))
        } else {
            None
        };
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn fwd(&self, x: &Tensor) -> Tensor {
        let x = x.conv_transpose2d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        );
        match &self.bias {
            Some(bias) => {
                let bias = bias.reshape([1, bias.shape_at(0), 1, 1]);
                x + bias
            }
            None => x,
        }
    }
}
