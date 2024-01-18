use std::collections::HashMap;

use crate::{non_differentiable, Tensor, TensorIter};

pub trait ValuAssociated {
    type ValueType;
    type Tensors: TensorIter + 'static;
    type Gradient;
}

impl<'a, T> ValuAssociated for &'a T
where
    T: ValuAssociated,
{
    type ValueType = T::ValueType;
    type Tensors = T::Tensors;
    type Gradient = T::Gradient;
}

pub trait Value: ValuAssociated {
    fn tensors(&self) -> Self::Tensors;
    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient;
    fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>);
}

pub trait VF<F, T, G> {
    fn vf_tensors(&self) -> T;

    fn vf_grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G;

    fn vf_grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>);
}

impl<T> Value for T
where
    T: ValuAssociated + VF<T::ValueType, T::Tensors, T::Gradient>,
{
    fn tensors(&self) -> Self::Tensors {
        <Self as VF<T::ValueType, T::Tensors, T::Gradient>>::vf_tensors(self)
    }

    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
        <Self as VF<T::ValueType, T::Tensors, T::Gradient>>::vf_grad(tensors, grad_map)
    }

    fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>) {
        <Self as VF<T::ValueType, T::Tensors, T::Gradient>>::vf_grad_map(tensors, grad, out);
    }
}

pub struct BasicType;

impl<'a, T, G, X> VF<BasicType, T, G> for &'a X
where
    X: VF<BasicType, T, G>,
{
    fn vf_tensors(&self) -> T {
        (*self).vf_tensors()
    }

    fn vf_grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G {
        X::vf_grad(tensors, grad_map)
    }

    fn vf_grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>) {
        X::vf_grad_map(tensors, grad, out)
    }
}

impl ValuAssociated for Tensor {
    type ValueType = BasicType;
    type Tensors = Tensor;
    type Gradient = Tensor;
}

impl VF<BasicType, Tensor, Tensor> for Tensor {
    fn vf_tensors(&self) -> Tensor {
        self.clone()
    }

    fn vf_grad(tensor: &Tensor, grad_map: &HashMap<usize, Tensor>) -> Tensor {
        grad_map.get(&tensor.id()).cloned().unwrap()
    }

    fn vf_grad_map(tensor: &Tensor, grad: Tensor, out: &mut HashMap<usize, Tensor>) {
        out.insert(tensor.id(), grad);
    }
}

impl ValuAssociated for HashMap<usize, Tensor> {
    type ValueType = BasicType;
    type Tensors = HashMap<usize, Tensor>;
    type Gradient = HashMap<usize, Tensor>;
}

impl VF<BasicType, HashMap<usize, Tensor>, HashMap<usize, Tensor>> for HashMap<usize, Tensor> {
    fn vf_tensors(&self) -> HashMap<usize, Tensor> {
        self.clone()
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

impl<const N: usize, T> ValuAssociated for [T; N]
where
    T: Value,
    [T::Tensors; N]: TensorIter,
{
    type ValueType = BasicType;
    type Tensors = [T::Tensors; N];
    type Gradient = [T::Gradient; N];
}

impl<const N: usize, T> VF<BasicType, [T::Tensors; N], [T::Gradient; N]> for [T; N]
where
    T: Value,
    [T::Tensors; N]: TensorIter,
{
    fn vf_tensors(&self) -> [T::Tensors; N] {
        self.iter()
            .map(Value::tensors)
            .collect::<Vec<T::Tensors>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    fn vf_grad(tensors: &[T::Tensors; N], grad_map: &HashMap<usize, Tensor>) -> [T::Gradient; N] {
        tensors
            .iter()
            .map(|t| T::grad(t, grad_map))
            .collect::<Vec<T::Gradient>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    fn vf_grad_map(
        tensors: &[T::Tensors; N],
        grad: [T::Gradient; N],
        out: &mut HashMap<usize, Tensor>,
    ) {
        for (t, g) in tensors.iter().zip(grad.into_iter()) {
            T::grad_map(t, g, out);
        }
    }
}

impl<A> ValuAssociated for (A,)
where
    A: Value,
    A::Tensors: TensorIter,
{
    type ValueType = BasicType;
    type Tensors = A::Tensors;
    type Gradient = A::Gradient;
}

impl<A> VF<BasicType, A::Tensors, A::Gradient> for (A,)
where
    A: Value,
    A::Tensors: TensorIter,
{
    fn vf_tensors(&self) -> A::Tensors {
        self.0.tensors()
    }

    fn vf_grad(tensors: &A::Tensors, grad_map: &HashMap<usize, Tensor>) -> A::Gradient {
        A::grad(tensors, grad_map)
    }

    fn vf_grad_map(tensors: &A::Tensors, grad: A::Gradient, out: &mut HashMap<usize, Tensor>) {
        A::grad_map(tensors, grad, out);
    }
}

macro_rules! impl_tuple_differentiable {
    ($($T:tt)*) => {
        paste::paste! {
            impl<$($T,)*> ValuAssociated for ($($T,)*)
            where
                $($T: Value,)*
                ($($T::Tensors,)*): TensorIter,
            {
                type ValueType = BasicType;
                type Tensors = ($($T::Tensors,)*);
                type Gradient = ($($T::Gradient,)*);
            }

            impl<$($T,)*> VF<BasicType, ($($T::Tensors,)*), ($($T::Gradient,)*)> for ($($T,)*)
            where
                $($T: Value,)*
                ($($T::Tensors,)*): TensorIter,
            {
                fn vf_tensors(&self) -> ($($T::Tensors,)*) {
                    let ($([<$T:lower 1>],)*) = self;
                    ($([<$T:lower 1>].tensors(),)*)
                }

                fn vf_grad(tensors: &($($T::Tensors,)*), grad_map: &HashMap<usize, Tensor>) -> ($($T::Gradient,)*) {
                    let ($([<$T:lower 1>],)*) = tensors;
                    ($($T::grad([<$T:lower 1>], grad_map),)*)
                }

                fn vf_grad_map(tensors: &($($T::Tensors,)*), grad: ($($T::Gradient,)*), out: &mut HashMap<usize, Tensor>) {
                    let ($([<$T:lower 1>],)*) = tensors;
                    let ($([<$T:lower 2>],)*) = grad;
                    $($T::grad_map([<$T:lower 1>], [<$T:lower 2>], out);)*
                }
            }
        }
    };
}

impl_tuple_differentiable!(A B);
impl_tuple_differentiable!(A B C);
impl_tuple_differentiable!(A B C D);
impl_tuple_differentiable!(A B C D E);
impl_tuple_differentiable!(A B C D E F);
impl_tuple_differentiable!(A B C D E F G);
impl_tuple_differentiable!(A B C D E F G H);
impl_tuple_differentiable!(A B C D E F G H I);
impl_tuple_differentiable!(A B C D E F G H I J);
impl_tuple_differentiable!(A B C D E F G H I J K);
impl_tuple_differentiable!(A B C D E F G H I J K L);

pub struct Aux<T>(pub T);
non_differentiable!(<T> Aux<T>);
