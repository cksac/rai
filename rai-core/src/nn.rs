use std::collections::HashMap;

use crate::{
    transforms::{ValuAssociated, VF},
    Tensor,
};

pub struct ModuleType;

pub trait Module {
    type Input;
    type Output;
    fn forward(&self, x: &Self::Input) -> Self::Output;
}

impl<'a, T> Module for &'a T
where
    T: Module,
{
    type Input = T::Input;
    type Output = T::Output;

    fn forward(&self, x: &Self::Input) -> Self::Output {
        (*self).forward(x)
    }
}

pub trait TrainableModule:
    Module
    + ValuAssociated<
        ValueType = ModuleType,
        Tensors = HashMap<usize, Tensor>,
        Gradient = HashMap<usize, Tensor>,
    >
{
    fn gather_params(&self, params: &mut HashMap<usize, Tensor>);

    fn params(&self) -> HashMap<usize, Tensor> {
        let mut params = HashMap::new();
        self.gather_params(&mut params);
        params
    }

    // TODO: params should be a reference? for model with shared parameters in different layers
    fn update_params(&self, params: &mut HashMap<usize, Tensor>);
}

impl<'a, T> TrainableModule for &'a T
where
    T: TrainableModule,
{
    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).gather_params(params)
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).update_params(params)
    }
}

impl<T> VF<ModuleType, HashMap<usize, Tensor>, HashMap<usize, Tensor>> for T
where
    T: TrainableModule<Tensors = HashMap<usize, Tensor>, Gradient = HashMap<usize, Tensor>>,
{
    fn vf_tensors(&self) -> HashMap<usize, Tensor> {
        self.params()
    }

    fn vf_grad(
        tensors: &HashMap<usize, Tensor>,
        grad_map: &HashMap<usize, Tensor>,
    ) -> HashMap<usize, Tensor> {
        tensors
            .keys()
            .map(|id| (*id, grad_map.get(id).unwrap().clone()))
            .collect()
    }

    fn vf_grad_map(
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
    Module + ValuAssociated<ValueType = ModuleType, Tensors = (), Gradient = ()>
{
}
impl<'a, T> NonTrainableModule for &'a T where T: NonTrainableModule {}

impl<T> VF<ModuleType, (), ()> for T
where
    T: NonTrainableModule<Tensors = (), Gradient = ()>,
{
    fn vf_tensors(&self) {}
    fn vf_grad(_: &(), _: &HashMap<usize, Tensor>) {}
    fn vf_grad_map(_: &(), _: (), _: &mut HashMap<usize, Tensor>) {}
}
