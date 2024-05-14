use crate::{non_differentiable, ty_kind, Tensor, TensorIter};
use half::{bf16, f16};
use std::collections::HashMap;
use ty_kind::Basic;

pub trait ValueSpec {
    type Kind;
    type Tensors: Clone + TensorIter + 'static;
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

impl<'a, T> ValueSpec for &'a mut T
where
    T: ValueSpec,
{
    type Kind = T::Kind;
    type Tensors = T::Tensors;
    type Gradient = T::Gradient;
}

pub trait Value: ValueSpec {
    fn to_tensor_vec(&self) -> Vec<Tensor> {
        self.tensors().tensor_iter().cloned().collect()
    }
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

impl<'a, T, G, X> GenericValue<ty_kind::Basic, T, G> for &'a X
where
    X: GenericValue<ty_kind::Basic, T, G>,
{
    #[inline]
    fn gv_tensors(&self) -> T {
        <X as GenericValue<ty_kind::Basic, T, G>>::gv_tensors(*self)
    }

    #[inline]
    fn gv_grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G {
        <X as GenericValue<ty_kind::Basic, T, G>>::gv_grad(tensors, grad_map)
    }

    #[inline]
    fn gv_grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>) {
        <X as GenericValue<ty_kind::Basic, T, G>>::gv_grad_map(tensors, grad, out)
    }
}

impl<'a, T, G, X> GenericValue<ty_kind::Basic, T, G> for &'a mut X
where
    X: GenericValue<ty_kind::Basic, T, G>,
{
    #[inline]
    fn gv_tensors(&self) -> T {
        <X as GenericValue<ty_kind::Basic, T, G>>::gv_tensors(*self)
    }

    #[inline]
    fn gv_grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G {
        <X as GenericValue<ty_kind::Basic, T, G>>::gv_grad(tensors, grad_map)
    }

    #[inline]
    fn gv_grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>) {
        <X as GenericValue<ty_kind::Basic, T, G>>::gv_grad_map(tensors, grad, out)
    }
}

impl<'a, A, T, G, X> GenericValue<ty_kind::Array<A>, T, G> for &'a X
where
    X: GenericValue<ty_kind::Array<A>, T, G>,
{
    #[inline]
    fn gv_tensors(&self) -> T {
        <X as GenericValue<ty_kind::Array<A>, T, G>>::gv_tensors(*self)
    }

    #[inline]
    fn gv_grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G {
        <X as GenericValue<ty_kind::Array<A>, T, G>>::gv_grad(tensors, grad_map)
    }

    #[inline]
    fn gv_grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>) {
        <X as GenericValue<ty_kind::Array<A>, T, G>>::gv_grad_map(tensors, grad, out)
    }
}

impl<'a, A, T, G, X> GenericValue<ty_kind::Tuple<A>, T, G> for &'a X
where
    X: GenericValue<ty_kind::Tuple<A>, T, G>,
{
    #[inline]
    fn gv_tensors(&self) -> T {
        <X as GenericValue<ty_kind::Tuple<A>, T, G>>::gv_tensors(*self)
    }

    #[inline]
    fn gv_grad(tensors: &T, grad_map: &HashMap<usize, Tensor>) -> G {
        <X as GenericValue<ty_kind::Tuple<A>, T, G>>::gv_grad(tensors, grad_map)
    }

    #[inline]
    fn gv_grad_map(tensors: &T, grad: G, out: &mut HashMap<usize, Tensor>) {
        <X as GenericValue<ty_kind::Tuple<A>, T, G>>::gv_grad_map(tensors, grad, out)
    }
}

impl ValueSpec for Tensor {
    type Kind = ty_kind::Basic;
    type Tensors = Tensor;
    type Gradient = Tensor;
}

impl GenericValue<ty_kind::Basic, Tensor, Tensor> for Tensor {
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
    type Kind = ty_kind::Basic;
    type Tensors = HashMap<usize, Tensor>;
    type Gradient = HashMap<usize, Tensor>;
}

impl GenericValue<ty_kind::Basic, HashMap<usize, Tensor>, HashMap<usize, Tensor>>
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
    type Kind = ty_kind::Array<[T; N]>;
    type Tensors = [T::Tensors; N];
    type Gradient = [T::Gradient; N];
}

impl<const N: usize, T> GenericValue<ty_kind::Array<[T; N]>, [T::Tensors; N], [T::Gradient; N]>
    for [T; N]
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
    type Kind = ty_kind::Tuple<(A,)>;
    type Tensors = A::Tensors;
    type Gradient = A::Gradient;
}

impl<A> GenericValue<ty_kind::Tuple<(A,)>, A::Tensors, A::Gradient> for (A,)
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
                type Kind = ty_kind::Tuple<($($T,)*)>;
                type Tensors = ($($T::Tensors,)*);
                type Gradient = ($($T::Gradient,)*);
            }

            impl<$($T,)*> GenericValue<ty_kind::Tuple<($($T,)*)>, ($($T::Tensors,)*), ($($T::Gradient,)*)> for ($($T,)*)
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

macro_rules! non_differentiable_primitive {
    ($T:ty) => {
        non_differentiable!(Basic; $T);
        non_differentiable!(Basic; Option<$T>);
        non_differentiable!(Basic; <E> Result<$T, E>);
    };
}

non_differentiable_primitive!(());
non_differentiable_primitive!(i8);
non_differentiable_primitive!(i16);
non_differentiable_primitive!(i32);
non_differentiable_primitive!(i64);
non_differentiable_primitive!(i128);
non_differentiable_primitive!(isize);
non_differentiable_primitive!(u8);
non_differentiable_primitive!(u16);
non_differentiable_primitive!(u32);
non_differentiable_primitive!(u64);
non_differentiable_primitive!(u128);
non_differentiable_primitive!(usize);
non_differentiable_primitive!(f16);
non_differentiable_primitive!(bf16);
non_differentiable_primitive!(f32);
non_differentiable_primitive!(f64);
non_differentiable_primitive!(bool);

pub struct Aux<T>(pub T);
non_differentiable!(Basic; <T> Aux<T>);
