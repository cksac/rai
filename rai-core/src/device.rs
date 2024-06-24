use crate::{ops::ToDevice, utils::cuda_enabled, Op, Tensor};
use std::{any::Any, borrow::Cow, fmt::Debug};

pub trait Device: Debug {
    fn name(&self) -> Cow<'static, str>;
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn Device>;
    fn eq(&self, rhs: &dyn Device) -> bool;
    fn to_device_op(&self) -> Box<dyn Op>;
}

impl<'a, T> Device for &'a T
where
    T: Device,
{
    fn name(&self) -> Cow<'static, str> {
        T::name(*self)
    }

    fn as_any(&self) -> &dyn Any {
        Device::as_any(*self)
    }

    fn clone_boxed(&self) -> Box<dyn Device> {
        Device::clone_boxed(*self)
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        Device::eq(*self, rhs)
    }

    fn to_device_op(&self) -> Box<dyn Op> {
        Device::to_device_op(*self)
    }
}

impl<'a> Device for &'a dyn Device {
    fn name(&self) -> Cow<'static, str> {
        Device::name(*self)
    }

    fn as_any(&self) -> &dyn Any {
        Device::as_any(*self)
    }

    fn clone_boxed(&self) -> Box<dyn Device> {
        Device::clone_boxed(*self)
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        Device::eq(*self, rhs)
    }

    fn to_device_op(&self) -> Box<dyn Op> {
        Device::to_device_op(*self)
    }
}

impl Device for Box<dyn Device> {
    fn name(&self) -> Cow<'static, str> {
        self.as_ref().name()
    }

    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn clone_boxed(&self) -> Box<dyn Device> {
        self.as_ref().clone_boxed()
    }

    fn eq(&self, rhs: &dyn Device) -> bool {
        self.as_ref().eq(rhs)
    }

    fn to_device_op(&self) -> Box<dyn Op> {
        self.as_ref().to_device_op()
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
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("cpu")
    }

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

    fn to_device_op(&self) -> Box<dyn Op> {
        Box::new(ToDevice::new(*self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cuda(usize);

impl Cuda {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn id(&self) -> usize {
        self.0
    }
}

impl Device for Cuda {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("cuda")
    }

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

    fn to_device_op(&self) -> Box<dyn Op> {
        Box::new(ToDevice::new(*self))
    }
}

pub fn cuda_if_available(id: usize) -> Box<dyn Device> {
    if cuda_enabled() {
        Box::new(Cuda::new(id))
    } else {
        Box::new(Cpu)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Metal(usize);

impl Metal {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn id(&self) -> usize {
        self.0
    }
}

impl Device for Metal {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("metal")
    }

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

    fn to_device_op(&self) -> Box<dyn Op> {
        Box::new(ToDevice::new(*self))
    }
}
