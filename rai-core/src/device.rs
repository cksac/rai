use std::{any::Any, fmt::Debug};

pub trait Device: Debug {
    fn as_any(&self) -> &dyn Any;
    fn clone_boxed(&self) -> Box<dyn Device>;
    fn equal(&self, rhs: &dyn Device) -> bool;
}

impl<'a, T> From<&'a T> for Box<dyn Device>
where
    T: Clone + Device + 'static,
{
    fn from(t: &'a T) -> Self {
        Box::new(t.clone())
    }
}

impl<'a> From<&'a Box<dyn Device>> for Box<dyn Device> {
    fn from(t: &'a Box<dyn Device>) -> Self {
        t.clone()
    }
}

impl Clone for Box<dyn Device> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl<'a> From<&'a dyn Device> for Box<dyn Device> {
    fn from(t: &'a dyn Device) -> Self {
        t.clone_boxed()
    }
}

impl<'a> PartialEq for &'a dyn Device {
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(*rhs)
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

    fn equal(&self, rhs: &dyn Device) -> bool {
        rhs.as_any().is::<Self>()
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

    fn equal(&self, rhs: &dyn Device) -> bool {
        rhs.as_any()
            .downcast_ref()
            .map_or(false, |other| self == other)
    }
}
