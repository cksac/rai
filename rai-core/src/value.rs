use crate::{non_differentiable, Tensor, TensorIter};
use std::collections::HashMap;

pub struct BasicValue;
pub struct ModuleValue;

pub trait ValueSpec {
    type Kind;
    type Tensors: TensorIter + 'static;
    type Gradient;
}

impl<'a, T> ValueSpec for &'a T
where
    T: ValueSpec,
{
    type Kind = T::Kind;
    type Tensors = T::Tensors;
    type Gradient = T::Gradient;
}

pub trait Value: ValueSpec {
    fn tensors(&self) -> Self::Tensors;
    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient;
    fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>);
}

pub trait GenericValue<Kind, Tensors, Gradient> {
    fn gv_tensors(&self) -> Tensors;

    fn gv_grad(tensors: &Tensors, grad_map: &HashMap<usize, Tensor>) -> Gradient;

    fn gv_grad_map(tensors: &Tensors, grad: Gradient, out: &mut HashMap<usize, Tensor>);
}

impl<T> Value for T
where
    T: ValueSpec + GenericValue<T::Kind, T::Tensors, T::Gradient>,
{
    #[inline]
    fn tensors(&self) -> Self::Tensors {
        <T as GenericValue<T::Kind, T::Tensors, T::Gradient>>::gv_tensors(self)
    }

    #[inline]
    fn grad(tensors: &Self::Tensors, grad_map: &HashMap<usize, Tensor>) -> Self::Gradient {
        <T as GenericValue<T::Kind, T::Tensors, T::Gradient>>::gv_grad(tensors, grad_map)
    }

    #[inline]
    fn grad_map(tensors: &Self::Tensors, grad: Self::Gradient, out: &mut HashMap<usize, Tensor>) {
        <T as GenericValue<T::Kind, T::Tensors, T::Gradient>>::gv_grad_map(tensors, grad, out);
    }
}

impl<'a, T, G, X> GenericValue<BasicValue, T, G> for &'a X
where
    X: GenericValue<BasicValue, T, G>,
{
    #[inline]
    fn gv_tensors(&self) -> T {
        (*self).gv_tensors()
    }

    #[inline]
    fn gv_grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G {
        <X as GenericValue<BasicValue, T, G>>::gv_grad(tensors, grad_map)
    }

    #[inline]
    fn gv_grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>) {
        <X as GenericValue<BasicValue, T, G>>::gv_grad_map(tensors, grad, out)
    }
}

impl ValueSpec for Tensor {
    type Kind = BasicValue;
    type Tensors = Tensor;
    type Gradient = Tensor;
}

impl GenericValue<BasicValue, Tensor, Tensor> for Tensor {
    fn gv_tensors(&self) -> Tensor {
        self.clone()
    }

    fn gv_grad(tensor: &Tensor, grad_map: &HashMap<usize, Tensor>) -> Tensor {
        grad_map.get(&tensor.id()).cloned().unwrap()
    }

    fn gv_grad_map(tensor: &Tensor, grad: Tensor, out: &mut HashMap<usize, Tensor>) {
        out.insert(tensor.id(), grad);
    }
}

impl ValueSpec for HashMap<usize, Tensor> {
    type Kind = BasicValue;
    type Tensors = HashMap<usize, Tensor>;
    type Gradient = HashMap<usize, Tensor>;
}

impl GenericValue<BasicValue, HashMap<usize, Tensor>, HashMap<usize, Tensor>>
    for HashMap<usize, Tensor>
{
    fn gv_tensors(&self) -> HashMap<usize, Tensor> {
        self.clone()
    }

    fn gv_grad(
        tensors: &HashMap<usize, Tensor>,
        grad_map: &HashMap<usize, Tensor>,
    ) -> HashMap<usize, Tensor> {
        tensors
            .keys()
            .map(|id| (*id, grad_map.get(id).unwrap().clone()))
            .collect()
    }

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

impl<const N: usize, T> ValueSpec for [T; N]
where
    T: Value,
    [T::Tensors; N]: TensorIter,
{
    type Kind = BasicValue;
    type Tensors = [T::Tensors; N];
    type Gradient = [T::Gradient; N];
}

impl<const N: usize, T> GenericValue<BasicValue, [T::Tensors; N], [T::Gradient; N]> for [T; N]
where
    T: Value,
    [T::Tensors; N]: TensorIter,
{
    fn gv_tensors(&self) -> [T::Tensors; N] {
        self.iter()
            .map(Value::tensors)
            .collect::<Vec<T::Tensors>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    fn gv_grad(tensors: &[T::Tensors; N], grad_map: &HashMap<usize, Tensor>) -> [T::Gradient; N] {
        tensors
            .iter()
            .map(|t| T::grad(t, grad_map))
            .collect::<Vec<T::Gradient>>()
            .try_into()
            .unwrap_or_else(|_| unreachable!())
    }

    fn gv_grad_map(
        tensors: &[T::Tensors; N],
        grad: [T::Gradient; N],
        out: &mut HashMap<usize, Tensor>,
    ) {
        for (t, g) in tensors.iter().zip(grad.into_iter()) {
            T::grad_map(t, g, out);
        }
    }
}

impl<A> ValueSpec for (A,)
where
    A: Value,
    A::Tensors: TensorIter,
{
    type Kind = BasicValue;
    type Tensors = A::Tensors;
    type Gradient = A::Gradient;
}

impl<A> GenericValue<BasicValue, A::Tensors, A::Gradient> for (A,)
where
    A: Value,
    A::Tensors: TensorIter,
{
    fn gv_tensors(&self) -> A::Tensors {
        self.0.tensors()
    }

    fn gv_grad(tensors: &A::Tensors, grad_map: &HashMap<usize, Tensor>) -> A::Gradient {
        A::grad(tensors, grad_map)
    }

    fn gv_grad_map(tensors: &A::Tensors, grad: A::Gradient, out: &mut HashMap<usize, Tensor>) {
        A::grad_map(tensors, grad, out);
    }
}

macro_rules! impl_tuple_differentiable {
    ($($T:tt)*) => {
        paste::paste! {
            impl<$($T,)*> ValueSpec for ($($T,)*)
            where
                $($T: Value,)*
                ($($T::Tensors,)*): TensorIter,
            {
                type Kind = BasicValue;
                type Tensors = ($($T::Tensors,)*);
                type Gradient = ($($T::Gradient,)*);
            }

            impl<$($T,)*> GenericValue<BasicValue, ($($T::Tensors,)*), ($($T::Gradient,)*)> for ($($T,)*)
            where
                $($T: Value,)*
                ($($T::Tensors,)*): TensorIter,
            {
                fn gv_tensors(&self) -> ($($T::Tensors,)*) {
                    let ($([<$T:lower 1>],)*) = self;
                    ($([<$T:lower 1>].tensors(),)*)
                }

                fn gv_grad(tensors: &($($T::Tensors,)*), grad_map: &HashMap<usize, Tensor>) -> ($($T::Gradient,)*) {
                    let ($([<$T:lower 1>],)*) = tensors;
                    ($($T::grad([<$T:lower 1>], grad_map),)*)
                }

                fn gv_grad_map(tensors: &($($T::Tensors,)*), grad: ($($T::Gradient,)*), out: &mut HashMap<usize, Tensor>) {
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
non_differentiable!(BasicValue; <T> Aux<T>);
