use crate::Tensor;
use std::{collections::HashMap, iter, ops::Deref};

pub trait TensorIter {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor>;
}

impl<'a, T> TensorIter for &'a T
where
    T: TensorIter,
{
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        (*self).tensor_iter()
    }
}

impl TensorIter for () {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        iter::empty()
    }
}

impl TensorIter for Tensor {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        iter::once(self)
    }
}

impl TensorIter for HashMap<usize, Tensor> {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.values()
    }
}

impl<const N: usize> TensorIter for [Tensor; N] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter()
    }
}

impl<'a, const N: usize> TensorIter for [&'a Tensor; N] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter().map(Deref::deref)
    }
}

macro_rules! impl_tuple_tensor_iter {
    ($($T:tt)*) => {
        paste::paste! {
            impl<$($T,)*> TensorIter for ($($T,)*)
            where
                $($T: TensorIter,)*
            {
                fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
                    let ($([<$T:lower 1>],)*) = self;
                    iter::empty()$(.chain([<$T:lower 1>].tensor_iter()))*
                }
            }
        }
    };
}

impl_tuple_tensor_iter!(A);
impl_tuple_tensor_iter!(A B);
impl_tuple_tensor_iter!(A B C);
impl_tuple_tensor_iter!(A B C D);
impl_tuple_tensor_iter!(A B C D E);
impl_tuple_tensor_iter!(A B C D E F);
impl_tuple_tensor_iter!(A B C D E F G);
impl_tuple_tensor_iter!(A B C D E F G H);
impl_tuple_tensor_iter!(A B C D E F G H I);
impl_tuple_tensor_iter!(A B C D E F G H I J);
impl_tuple_tensor_iter!(A B C D E F G H I J K);
impl_tuple_tensor_iter!(A B C D E F G H I J K L);
