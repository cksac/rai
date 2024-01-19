use std::{collections::HashMap, path::Path};

use safetensors::SafeTensors;

use crate::{backend::Cpu, Backend, GenericValue, ModuleValue, Tensor, ValueSpec};

pub trait Module {
    type Input;
    type Output;
    fn forward(&self, x: &Self::Input) -> Self::Output;

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>);

    fn params(&self) -> HashMap<usize, Tensor> {
        let mut params = HashMap::new();
        self.gather_params(&mut params);
        params
    }

    // TODO: params should be a reference? for model with shared parameters in different layers
    fn update_params(&self, params: &mut HashMap<usize, Tensor>);

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>);

    fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        self.gather_named_params(prefix, &mut params);
        params
    }

    fn to_safetensors<P: AsRef<Path>>(&self, filename: P)
    where
        Self: Sized,
    {
        let named_params = self.named_params("");
        // todo: add to_device ops, and move all tensors to cpu first
        // todo: eval once after call to_device
        Cpu.to_safetensors(named_params, filename.as_ref());
    }
}

impl<'a, T> Module for &'a T
where
    T: Module,
{
    type Input = T::Input;
    type Output = T::Output;

    #[inline]
    fn forward(&self, x: &Self::Input) -> Self::Output {
        (*self).forward(x)
    }

    #[inline]
    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).gather_params(params)
    }

    #[inline]
    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).update_params(params)
    }

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        (*self).gather_named_params(prefix, params)
    }
}

pub trait TrainableModule:
    Module
    + ValueSpec<
        Kind = ModuleValue,
        Tensors = HashMap<usize, Tensor>,
        Gradient = HashMap<usize, Tensor>,
    >
{
}

impl<'a, T> TrainableModule for &'a T where T: TrainableModule {}

impl<T> GenericValue<ModuleValue, HashMap<usize, Tensor>, HashMap<usize, Tensor>> for T
where
    T: TrainableModule<Tensors = HashMap<usize, Tensor>, Gradient = HashMap<usize, Tensor>>,
{
    #[inline]
    fn gv_tensors(&self) -> HashMap<usize, Tensor> {
        self.params()
    }

    #[inline]
    fn gv_grad(
        tensors: &HashMap<usize, Tensor>,
        grad_map: &HashMap<usize, Tensor>,
    ) -> HashMap<usize, Tensor> {
        tensors
            .keys()
            .map(|id| (*id, grad_map.get(id).unwrap().clone()))
            .collect()
    }

    #[inline]
    fn gv_grad_map(
        tensors: &HashMap<usize, Tensor>,
        grad: HashMap<usize, Tensor>,
        out: &mut HashMap<usize, Tensor>,
    ) {
        for id in tensors.keys() {
            out.insert(*id, grad.get(id).unwrap().clone());
        }
    }
}

pub trait NonTrainableModule:
    Module + ValueSpec<Kind = ModuleValue, Tensors = (), Gradient = ()>
{
}
impl<'a, T> NonTrainableModule for &'a T where T: NonTrainableModule {}

impl<T> GenericValue<ModuleValue, (), ()> for T
where
    T: NonTrainableModule<Tensors = (), Gradient = ()>,
{
    fn gv_tensors(&self) {}
    fn gv_grad(_: &(), _: &HashMap<usize, Tensor>) {}
    fn gv_grad_map(_: &(), _: (), _: &mut HashMap<usize, Tensor>) {}
}
