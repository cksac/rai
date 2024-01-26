use crate::{primitives::ToDevice, Primitive, Tensor};
use std::{any::Any, fmt::Debug};

pub trait Device: Debug {
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn Device>;
    fn eq(&self, rhs: &dyn Device) -> bool;
    fn primitive_to_device(&self) -> Box<dyn Primitive>;
}

impl<'a> Device for &'a dyn Device {
    fn as_any(&self) -> &dyn Any {
        Device::as_any(*self)
    }

    fn clone_boxed(&self) -> Box<dyn Device> {
        Device::clone_boxed(*self)
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        Device::eq(*self, rhs)
    }

    fn primitive_to_device(&self) -> Box<dyn Primitive> {
        Device::primitive_to_device(*self)
    }
}

impl Clone for Box<dyn Device> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl PartialEq for Box<dyn Device> {
    fn eq(&self, rhs: &Self) -> bool {
        Device::eq(self.as_ref(), rhs.as_ref())
    }
}

impl<'a> PartialEq for &'a dyn Device {
    fn eq(&self, rhs: &Self) -> bool {
        Device::eq(self, rhs)
    }
}

pub trait AsDevice: Debug {
    fn device(&self) -> &dyn Device;
    fn into_boxed_device(self) -> Box<dyn Device>;
}

impl<D: Device> AsDevice for D {
    fn device(&self) -> &dyn Device {
        self as &dyn Device
    }

    fn into_boxed_device(self) -> Box<dyn Device> {
        self.clone_boxed()
    }
}

impl AsDevice for Tensor {
    fn device(&self) -> &dyn Device {
        Tensor::device(self)
    }

    fn into_boxed_device(self) -> Box<dyn Device> {
        Tensor::device(&self).clone_boxed()
    }
}

impl<'a> AsDevice for &'a Tensor {
    fn device(&self) -> &dyn Device {
        Tensor::device(self)
    }

    fn into_boxed_device(self) -> Box<dyn Device> {
        Tensor::device(self).clone_boxed()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cpu;

impl Device for Cpu {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn Device> {
        Box::new(*self)
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        rhs.as_any()
            .downcast_ref::<Self>()
            .map_or(false, |other| self == other)
    }

    fn primitive_to_device(&self) -> Box<dyn Primitive> {
        Box::new(ToDevice::new(*self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cuda(pub usize);

impl Device for Cuda {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn Device> {
        Box::new(*self)
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        rhs.as_any()
            .downcast_ref::<Self>()
            .map_or(false, |other| self == other)
    }

    fn primitive_to_device(&self) -> Box<dyn Primitive> {
        Box::new(ToDevice::new(*self))
    }
}
