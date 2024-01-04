use std::{
    any::{Any, TypeId},
    fmt::Debug,
};

pub trait Backend: Debug {
    fn clone_boxed(&self) -> Box<dyn Backend>;
    fn data_type_id(&self) -> TypeId;
    fn as_any(&self) -> &dyn Any;
}

impl<T> From<&T> for Box<dyn Backend>
where
    T: Clone + Backend + 'static,
{
    fn from(t: &T) -> Self {
        Box::new(t.clone())
    }
}

impl From<&Box<dyn Backend>> for Box<dyn Backend> {
    fn from(t: &Box<dyn Backend>) -> Self {
        t.clone()
    }
}

impl Clone for Box<dyn Backend> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl From<&dyn Backend> for Box<dyn Backend> {
    fn from(t: &dyn Backend) -> Self {
        t.clone_boxed()
    }
}

mod cpu;
pub use cpu::Cpu;

// mod cuda;
// pub use cuda::Cuda;
