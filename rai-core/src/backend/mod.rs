use dyn_clone::DynClone;
use std::{
    any::{Any, TypeId},
    fmt::Debug,
};

pub trait Backend: Debug + DynClone {
    fn data_type_id(&self) -> TypeId;
    fn as_any(&self) -> &dyn Any;
}

dyn_clone::clone_trait_object!(Backend);

impl From<&Box<dyn Backend>> for Box<dyn Backend> {
    fn from(value: &Box<dyn Backend>) -> Self {
        value.clone()
    }
}

impl<T> From<&T> for Box<dyn Backend>
where
    T: Clone + Backend + 'static,
{
    fn from(t: &T) -> Self {
        Box::new(t.clone())
    }
}

mod cpu;
pub use cpu::Cpu;

// mod cuda;
// pub use cuda::Cuda;
