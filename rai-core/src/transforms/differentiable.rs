use std::collections::HashMap;

use crate::{non_differentiable, Tensor, TensorIter};

pub trait Differentiable {
    type Tensors: TensorIter + 'static;
    type Gradient;
    fn tensors(&self) -> Self::Tensors;
    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient;
    fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>);
}

// impl<'a, T> Differentiable for &'a T
// where
//     T: Differentiable,
// {
//     type Tensors = T::Tensors;
//     type Gradient = T::Gradient;

//     fn tensors(&self) -> Self::Tensors {
//         (*self).tensors()
//     }

//     fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
//         T::grad(tensors, grad_map)
//     }

//     fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>) {
//         T::grad_map(tensors, grad, out)
//     }
// }

impl Differentiable for Tensor {
    type Tensors = Tensor;
    type Gradient = Tensor;

    fn tensors(&self) -> Self::Tensors {
        self.clone()
    }

    fn grad(tensor: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
        grad_map.get(&tensor.id()).cloned().unwrap()
    }

    fn grad_map(tensor: &Self::Tensors, grad: Tensor, out: &mut HashMap<usize, Tensor>) {
        out.insert(tensor.id(), grad);
    }
}

impl<'a> Differentiable for &'a Tensor {
    type Tensors = Tensor;
    type Gradient = Tensor;

    fn tensors(&self) -> Self::Tensors {
        (*self).clone()
    }

    fn grad(tensor: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
        grad_map.get(&tensor.id()).cloned().unwrap()
    }

    fn grad_map(tensor: &Self::Tensors, grad: Tensor, out: &mut HashMap<usize, Tensor>) {
        out.insert(tensor.id(), grad);
    }
}

impl Differentiable for HashMap<usize, Tensor> {
    type Tensors = HashMap<usize, Tensor>;
    type Gradient = HashMap<usize, Tensor>;

    fn tensors(&self) -> Self::Tensors {
        self.clone()
    }

    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
        tensors
            .keys()
            .map(|id| (*id, grad_map.get(id).unwrap().clone()))
            .collect()
    }

    fn grad_map(
        tensors: &Self::Tensors,
        grad: HashMap<usize, Tensor>,
        out: &mut HashMap<usize, Tensor>,
    ) {
        for id in tensors.keys() {
            out.insert(*id, grad.get(id).unwrap().clone());
        }
    }
}

impl<const N: usize, T> Differentiable for [T; N]
where
    T: Differentiable,
    [T::Tensors; N]: TensorIter,
{
    type Tensors = [T::Tensors; N];

    type Gradient = [T::Gradient; N];

    fn tensors(&self) -> Self::Tensors {
        self.iter()
            .map(Differentiable::tensors)
            .collect::<Vec<T::Tensors>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
        tensors
            .iter()
            .map(|t| T::grad(t, grad_map))
            .collect::<Vec<T::Gradient>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>) {
        for (t, g) in tensors.iter().zip(grad.into_iter()) {
            T::grad_map(t, g, out);
        }
    }
}

impl<A> Differentiable for (A,)
where
    A: Differentiable,
    A::Tensors: TensorIter,
{
    type Tensors = A::Tensors;
    type Gradient = A::Gradient;

    fn tensors(&self) -> Self::Tensors {
        self.0.tensors()
    }

    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
        A::grad(tensors, grad_map)
    }

    fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>) {
        A::grad_map(tensors, grad, out);
    }
}

macro_rules! impl_tuple_differentiable {
    ($($T:tt)*) => {
        paste::paste! {
            impl<$($T,)*> Differentiable for ($($T,)*)
            where
                $($T: Differentiable,)*
                ($($T::Tensors,)*): TensorIter,
            {
                type Tensors = ($($T::Tensors,)*);
                type Gradient = ($($T::Gradient,)*);

                fn tensors(&self) -> Self::Tensors {
                    let ($([<$T:lower 1>],)*) = self;
                    ($([<$T:lower 1>].tensors(),)*)
                }

                fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
                    let ($([<$T:lower 1>],)*) = tensors;
                    ($($T::grad([<$T:lower 1>], grad_map),)*)
                }

                fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>) {
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
