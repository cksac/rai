use std::{
    any::{Any, TypeId},
    fmt::Debug,
};

pub trait Backend: Debug {
    fn clone_boxed(&self) -> Box<dyn Backend>;
    fn data_type_id(&self) -> TypeId;
    fn as_any(&self) -> &dyn Any;
    fn equal(&self, rhs: &dyn Backend) -> bool;
}

impl<'a, T> From<&'a T> for Box<dyn Backend>
where
    T: Clone + Backend + 'static,
{
    fn from(t: &'a T) -> Self {
        Box::new(t.clone())
    }
}

impl<'a> From<&'a Box<dyn Backend>> for Box<dyn Backend> {
    fn from(t: &'a Box<dyn Backend>) -> Self {
        t.clone()
    }
}

impl Clone for Box<dyn Backend> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl<'a> From<&'a dyn Backend> for Box<dyn Backend> {
    fn from(t: &'a dyn Backend) -> Self {
        t.clone_boxed()
    }
}

impl<'a> PartialEq for &'a dyn Backend {
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(*rhs)
    }
}

mod cpu;
pub use cpu::Cpu;

// mod cuda;
// pub use cuda::Cuda;
