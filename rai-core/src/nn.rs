use std::collections::HashMap;

use crate::{Differentiable, Tensor, TensorIter};

pub trait Module {
    type Tensors: TensorIter + 'static;
    type Gradient;

    type Input<'i>;
    type Output<'o>;
    fn forward<'i, 'o>(&self, x: Self::Input<'i>) -> Self::Output<'o>;
}

impl<'a, T> Module for &'a T
where
    T: Module,
{
    type Tensors = T::Tensors;

    type Gradient = T::Gradient;

    type Input<'i> = T::Input<'i>;

    type Output<'o> = T::Output<'o>;

    fn forward<'i, 'o>(&self, x: Self::Input<'i>) -> Self::Output<'o> {
        (*self).forward(x)
    }
}

pub trait TrainableModule:
    Module<Tensors = HashMap<usize, Tensor>, Gradient = HashMap<usize, Tensor>>
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
        (*self).gather_params(params);
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).update_params(params);
    }
}

pub trait NonTrainableModule: Module<Tensors = (), Gradient = ()> {}
impl<'a, T> NonTrainableModule for &'a T where T: NonTrainableModule {}

mod private {
    use super::*;

    impl<T: Module + DifferentiableModule<T::Tensors, T::Gradient>> Differentiable for T {
        type Tensors = T::Tensors;
        type Gradient = T::Gradient;

        fn tensors(&self) -> Self::Tensors {
            <Self as DifferentiableModule<T::Tensors, T::Gradient>>::tensors(self)
        }

        fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
            <Self as DifferentiableModule<T::Tensors, T::Gradient>>::grad(tensors, grad_map)
        }

        fn grad_map(
            tensors: &Self::Tensors,
            grad: Self::Gradient,
            out: &mut HashMap<usize, Tensor>,
        ) {
            <Self as DifferentiableModule<T::Tensors, T::Gradient>>::grad_map(tensors, grad, out)
        }
    }

    trait DifferentiableModule<T, G> {
        fn tensors(&self) -> T;

        fn grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G;

        fn grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>);
    }

    impl<T: TrainableModule> DifferentiableModule<HashMap<usize, Tensor>, HashMap<usize, Tensor>>
        for T
    {
        fn tensors(&self) -> HashMap<usize, Tensor> {
            TrainableModule::params(self)
        }

        fn grad(
            tensors: &HashMap<usize, Tensor>,
            grad_map: &HashMap<usize, Tensor>,
        ) -> HashMap<usize, Tensor> {
            tensors
                .keys()
                .map(|id| (*id, grad_map.get(id).unwrap().clone()))
                .collect()
        }

        fn grad_map(
            tensors: &HashMap<usize, Tensor>,
            grad: HashMap<usize, Tensor>,
            out: &mut HashMap<usize, Tensor>,
        ) {
            for id in tensors.keys() {
                out.insert(*id, grad.get(id).unwrap().clone());
            }
        }
    }

    impl<T: NonTrainableModule> DifferentiableModule<(), ()> for T {
        fn tensors(&self) {}
        fn grad(_: &(), _: &HashMap<usize, Tensor>) {}
        fn grad_map(_: &(), _: (), _: &mut HashMap<usize, Tensor>) {}
    }
}
