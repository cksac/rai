use crate::Shape;
use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

pub trait Dim: Debug {
    fn dim_of<S: Shape + ?Sized>(&self, shape: &S) -> usize;
    fn dim_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> usize;
    fn size_of<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        let dim = self.dim_of(shape);
        shape.shape()[dim]
    }
    fn size_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        let dim = self.dim_of_unchecked(shape);
        shape.shape()[dim]
    }
    fn is_negative(&self) -> bool;
}

impl<'a, D> Dim for &'a D
where
    D: Dim,
{
    fn dim_of<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        (*self).dim_of(shape)
    }

    fn dim_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        (*self).dim_of_unchecked(shape)
    }

    fn is_negative(&self) -> bool {
        (*self).is_negative()
    }
}

impl Dim for isize {
    fn dim_of<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        let dim = if self.is_negative() {
            shape.rank() - self.unsigned_abs()
        } else {
            *self as usize
        };
        assert!(dim < shape.rank(), "dimension out of bounds");
        dim
    }

    fn dim_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        let dim = if self.is_negative() {
            shape.rank() - self.unsigned_abs()
        } else {
            *self as usize
        };
        debug_assert!(dim < shape.rank(), "dimension out of bounds");
        dim
    }

    fn is_negative(&self) -> bool {
        *self < 0
    }
}

impl Dim for i32 {
    fn dim_of<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        let dim = if *self < 0 {
            shape.rank() - self.unsigned_abs() as usize
        } else {
            *self as usize
        };
        assert!(dim < shape.rank(), "dimension out of bounds");
        dim
    }

    fn dim_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        let dim = if *self < 0 {
            shape.rank() - self.unsigned_abs() as usize
        } else {
            *self as usize
        };
        debug_assert!(dim < shape.rank(), "dimension out of bounds");
        dim
    }

    fn is_negative(&self) -> bool {
        *self < 0
    }
}

impl Dim for usize {
    fn dim_of<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        assert!(*self < shape.rank(), "dimension out of bounds");
        *self
    }

    fn dim_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> usize {
        debug_assert!(*self < shape.rank(), "dimension out of bounds");
        *self
    }

    fn is_negative(&self) -> bool {
        false
    }
}

pub trait Dims<Output>: Debug {
    fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Output;
    fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Output;
    fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Output;
    fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Output;
}

impl<'a, O, D> Dims<O> for &'a D
where
    D: Dims<O>,
{
    fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> O {
        (*self).dims_of(shape)
    }

    fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> O {
        (*self).dims_of_unchecked(shape)
    }

    fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> O {
        (*self).sizes_of(shape)
    }

    fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> O {
        (*self).sizes_of_unchecked(shape)
    }
}

macro_rules! impl_dims_single {
    ($($t:ty),*) => {
        $(
            impl Dims<Vec<usize>> for $t {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let dim = self.dim_of(shape);
                    vec![dim]
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let dim = self.dim_of_unchecked(shape);
                    vec![dim]
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let size = self.size_of(shape);
                    vec![size]
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let size = self.size_of_unchecked(shape);
                    vec![size]
                }
            }

            impl Dims<[usize; 1]> for $t {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> [usize; 1] {
                    let dim = self.dim_of(shape);
                    [dim]
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> [usize; 1] {
                    let dim = self.dim_of_unchecked(shape);
                    [dim]
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> [usize; 1] {
                    let size = self.size_of(shape);
                    [size]
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> [usize; 1] {
                    let size = self.size_of_unchecked(shape);
                    [size]
                }
            }
        )*
    };
}

impl_dims_single!(isize, i32, usize);

macro_rules! impl_dims_vec {
    ($($t:ty),*) => {
        $(
            impl Dims<Vec<usize>> for Vec<$t> {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of(shape)).collect()
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of_unchecked(shape)).collect()
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of(shape)).collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of_unchecked(shape)).collect()
                }
            }
        )*
    };
}

impl_dims_vec!(isize, i32, usize);

macro_rules! impl_dims_array {
    ($($t:ty),*) => {
        $(
            impl<const N: usize> Dims<Vec<usize>> for [$t; N] {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of(shape)).collect()
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of_unchecked(shape)).collect()
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of(shape)).collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of_unchecked(shape)).collect()
                }
            }

            impl<const N: usize> Dims<[usize; N]> for [$t; N] {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N]{
                    self.iter().map(|&dim| dim.dim_of(shape)).collect::<Vec<_>>().try_into().unwrap()
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N] {
                    self.iter().map(|&dim| dim.dim_of_unchecked(shape)).collect::<Vec<_>>().try_into().unwrap()
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N] {
                    self.iter().map(|&dim| dim.size_of(shape)).collect::<Vec<_>>().try_into().unwrap()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N] {
                    self.iter().map(|&dim| dim.size_of_unchecked(shape)).collect::<Vec<_>>().try_into().unwrap()
                }
            }

            impl Dims<Vec<usize>> for [$t] {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of(shape)).collect()
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of_unchecked(shape)).collect()
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of(shape)).collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of_unchecked(shape)).collect()
                }
            }

            impl<'a> Dims<Vec<usize>> for &'a [$t] {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of(shape)).collect()
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.dim_of_unchecked(shape)).collect()
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of(shape)).collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.iter().map(|&dim| dim.size_of_unchecked(shape)).collect()
                }
            }
        )*
    };
}

impl_dims_array!(isize, i32, usize);

macro_rules! impl_dims_range {
    ($($t:ty),*) => {
        $(
            impl Dims<Vec<usize>> for Range<$t> {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let start = shape.dim(self.start);
                    let end = shape.dim(self.end);
                    if start > end {
                        (end + 1..=start).rev().collect()
                    } else {
                        (start..end).collect()
                    }
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let start = shape.dim_unchecked(self.start);
                    let end = shape.dim(self.end);
                    if start > end {
                        (end + 1..=start).rev().collect()
                    } else {
                        (start..end).collect()
                    }
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of_unchecked(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }
            }

            impl Dims<Vec<usize>> for RangeInclusive<$t> {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let start = shape.dim(*self.start());
                    let end = shape.dim(*self.end());
                    if start > end {
                        (end..=start).rev().collect()
                    } else {
                        (start..=end).collect()
                    }
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let start = shape.dim(*self.start());
                    let end = shape.dim(*self.end());
                    if start > end {
                        (end..=start).rev().collect()
                    } else {
                        (start..=end).collect()
                    }
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of_unchecked(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }
            }

            impl Dims<Vec<usize>> for RangeFrom<$t> {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let start = shape.dim(self.start);
                    let end = shape.rank();
                    if start > end {
                        (end + 1..=start).rev().collect()
                    } else {
                        (start..end).collect()
                    }
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let start = shape.dim_unchecked(self.start);
                    let end = shape.rank();
                    if start > end {
                        (end + 1..=start).rev().collect()
                    } else {
                        (start..end).collect()
                    }
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of_unchecked(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }
            }

            impl Dims<Vec<usize>> for RangeTo<$t> {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let dim = shape.dim(self.end);
                    (0..dim).collect()
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let dim = shape.dim_unchecked(self.end);
                    (0..dim).collect()
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of_unchecked(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }
            }

            impl Dims<Vec<usize>> for RangeToInclusive<$t> {
                fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let dim = shape.dim(self.end);
                    (0..=dim).collect()
                }

                fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    let dim = shape.dim_unchecked(self.end);
                    (0..=dim).collect()
                }

                fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }

                fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
                    self.dims_of_unchecked(shape)
                        .iter()
                        .map(|&dim| dim.size_of_unchecked(shape))
                        .collect()
                }
            }
        )*
    };
}

impl_dims_range!(isize, i32, usize);

impl Dims<Vec<usize>> for RangeFull {
    fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        (0..shape.rank()).collect()
    }

    fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        (0..shape.rank()).collect()
    }

    fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        shape.shape().to_vec()
    }

    fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        shape.shape().to_vec()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Before<const N: usize>;

impl<const N: usize> Dims<Vec<usize>> for Before<N> {
    fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        (0..N).map(|dim| dim.dim_of(shape)).collect()
    }

    fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        (0..N).map(|dim| dim.dim_of_unchecked(shape)).collect()
    }

    fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        (0..N).map(|dim| dim.size_of(shape)).collect()
    }

    fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> Vec<usize> {
        (0..N).map(|dim| dim.size_of_unchecked(shape)).collect()
    }
}

impl<const N: usize> Dims<[usize; N]> for Before<N> {
    fn dims_of<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N] {
        (0..N)
            .map(|dim| dim.dim_of(shape))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn dims_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N] {
        (0..N)
            .map(|dim| dim.dim_of_unchecked(shape))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn sizes_of<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N] {
        (0..N)
            .map(|dim| dim.size_of(shape))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn sizes_of_unchecked<S: Shape + ?Sized>(&self, shape: &S) -> [usize; N] {
        (0..N)
            .map(|dim| dim.size_of_unchecked(shape))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
