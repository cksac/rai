use crate::{AsDevice, Device, Op, Shape, Tensor};
use std::{any::Any, borrow::Cow};

#[derive(Clone, Debug)]
pub struct ToDevice<D: Device + Clone> {
    pub device: D,
}

impl<D: Device + Clone> ToDevice<D> {
    pub fn new(device: D) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &D {
        &self.device
    }
}

impl<D: Device + Clone + 'static> Op for ToDevice<D> {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(format!("ToDevice<{}>", self.device.name()))
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("ToDevice({:?})", self.device())
    }

    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.to_device(self.device())
    }

    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.to_device(x);
        vec![cotangent_x]
    }
}

#[track_caller]
pub fn to_device(x: &Tensor, device: impl AsDevice) -> Tensor {
    let device = device.device();
    if x.device() == device {
        return x.clone();
    }
    let dtype = x.dtype();
    let shape = x.shape().to_vec();
    let inputs = vec![x.clone()];
    let op = device.to_device_op();
    Tensor::new(device, dtype, shape, op, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn to_device(&self, device: impl AsDevice) -> Tensor {
        to_device(self, device)
    }
}
