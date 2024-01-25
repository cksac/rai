use crate::{primitives::ToDevice, Primitive, Tensor};
use std::{any::Any, fmt::Debug};

pub trait IDevice: Clone + Debug + PartialEq {
    type Repr: IDevice;
    fn as_any(&self) -> &dyn Any;
    fn eq(&self, rhs: &dyn Device) -> bool;
    fn primitive_to_device(&self) -> ToDevice<Self::Repr>;
}

impl<'a, D: IDevice> IDevice for &'a D {
    type Repr = D::Repr;

    fn as_any(&self) -> &dyn Any {
        IDevice::as_any(*self)
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        IDevice::eq(*self, rhs)
    }

    fn primitive_to_device(&self) -> ToDevice<Self::Repr> {
        IDevice::primitive_to_device(*self)
    }
}

pub trait Device: Debug {
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn Device>;
    fn eq(&self, rhs: &dyn Device) -> bool;
    fn primitive_to_device(&self) -> Box<dyn Primitive>;
}

impl<D: IDevice + 'static> Device for D {
    fn as_any(&self) -> &dyn Any {
        IDevice::as_any(self)
    }

    fn clone_boxed(&self) -> Box<dyn Device> {
        Box::new(self.clone())
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        IDevice::eq(self, rhs)
    }

    fn primitive_to_device(&self) -> Box<dyn Primitive> {
        Box::new(IDevice::primitive_to_device(self))
    }
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
    fn boxed_device(&self) -> Box<dyn Device>;
}

impl<D: Device> AsDevice for D {
    fn device(&self) -> &dyn Device {
        self as &dyn Device
    }

    fn boxed_device(&self) -> Box<dyn Device> {
        self.clone_boxed()
    }
}

impl AsDevice for Tensor {
    fn device(&self) -> &dyn Device {
        Tensor::device(self)
    }

    fn boxed_device(&self) -> Box<dyn Device> {
        Tensor::device(self).clone_boxed()
    }
}

impl<'a> AsDevice for &'a Tensor {
    fn device(&self) -> &dyn Device {
        Tensor::device(*self)
    }

    fn boxed_device(&self) -> Box<dyn Device> {
        Tensor::device(*self).clone_boxed()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cpu;

impl IDevice for Cpu {
    type Repr = Self;

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        rhs.as_any()
            .downcast_ref::<Self>()
            .map_or(false, |other| self == other)
    }

    fn primitive_to_device(&self) -> ToDevice<Self::Repr> {
        ToDevice::new(*self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cuda(pub usize);

impl IDevice for Cuda {
    type Repr = Self;

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        rhs.as_any()
            .downcast_ref::<Self>()
            .map_or(false, |other| self == other)
    }

    fn primitive_to_device(&self) -> ToDevice<Self::Repr> {
        ToDevice::new(*self)
    }
}
