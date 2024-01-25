use std::{any::Any, fmt::Debug};

use crate::{primitives::ToDevice, Primitive};

pub trait Device: Clone + Copy + Debug + PartialEq {
    fn primitive_to_device(&self) -> ToDevice<Self> {
        ToDevice::new(*self)
    }
}

impl<'a, T: Device> Device for &'a T {
    fn primitive_to_device(&self) -> ToDevice<Self> {
        ToDevice::new(*self)
    }
}

pub trait DynDevice: Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn DynDevice>;
    fn equal(&self, rhs: &dyn DynDevice) -> bool;
    fn primitive_to_device(&self) -> Box<dyn Primitive>;
}

impl<D: Device + 'static> DynDevice for D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn DynDevice> {
        Box::new(*self)
    }

    fn equal(&self, rhs: &dyn DynDevice) -> bool {
        rhs.as_any()
            .downcast_ref()
            .map_or(false, |other| self == other)
    }

    fn primitive_to_device(&self) -> Box<dyn Primitive> {
        Box::new(self.primitive_to_device())
    }
}

impl<'a, T> From<&'a T> for Box<dyn DynDevice>
where
    T: Clone + DynDevice + 'static,
{
    fn from(t: &'a T) -> Self {
        Box::new(t.clone())
    }
}

impl<'a> From<&'a Box<dyn DynDevice>> for Box<dyn DynDevice> {
    fn from(t: &'a Box<dyn DynDevice>) -> Self {
        t.clone()
    }
}

impl Clone for Box<dyn DynDevice> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl<'a> From<&'a dyn DynDevice> for Box<dyn DynDevice> {
    fn from(t: &'a dyn DynDevice) -> Self {
        t.clone_boxed()
    }
}

impl<'a> PartialEq for &'a dyn DynDevice {
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(*rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cpu;

impl Device for Cpu {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cuda(pub usize);

impl Device for Cuda {}
