use std::{
    fmt::Debug,
    ops::{RangeFull, RangeTo},
};

use crate::{Error, Result};

pub trait Dim: Debug {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize;
}

impl<'a, T> Dim for &'a T
where
    T: Dim + ?Sized,
{
    fn dim_of<U: Shape + ?Sized>(&self, shape: &U) -> usize {
        (*self).dim_of(shape)
    }
}

impl Dim for usize {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize {
        assert!(*self < shape.ndim(), "{} < {}", self, shape.ndim());
        *self
    }
}

impl Dim for isize {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize {
        let dim = if *self >= 0 {
            *self as usize
        } else {
            assert!(shape.ndim() > 0);
            self.checked_add_unsigned(shape.ndim() - 1).unwrap() as usize
        };
        assert!(dim < shape.ndim(), "{} < {}", dim, shape.ndim());
        dim
    }
}

impl Dim for i32 {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize {
        let dim = if *self >= 0 {
            *self as usize
        } else {
            assert!(shape.ndim() > 0);
            self.checked_add_unsigned(shape.ndim() as u32 - 1).unwrap() as usize
        };
        assert!(dim < shape.ndim(), "{} < {}", dim, shape.ndim());
        dim
    }
}

impl Dim for RangeFull {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize {
        assert!(shape.ndim() > 0);
        shape.ndim() - 1
    }
}

pub trait Dims: Debug {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize>;
}

impl<'a, T> Dims for &'a T
where
    T: Dims + ?Sized,
{
    fn dims_of<U: Shape + ?Sized>(&self, shape: &U) -> Vec<usize> {
        (*self).dims_of(shape)
    }
}

impl Dims for usize {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        vec![shape.dim(*self)]
    }
}

impl Dims for isize {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        vec![shape.dim(*self)]
    }
}

impl Dims for i32 {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        vec![shape.dim(*self)]
    }
}

impl Dims for Vec<usize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl Dims for Vec<isize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl Dims for Vec<i32> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl<const N: usize> Dims for [isize; N] {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl<const N: usize> Dims for [i32; N] {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl Dims for [usize] {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl Dims for [isize] {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl Dims for [i32] {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        self.iter().map(|d| shape.dim(*d)).collect()
    }
}

impl Dims for RangeFull {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        (0..shape.ndim()).collect()
    }
}

impl Dims for RangeTo<usize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let dim = shape.dim(self.end);
        (0..dim).collect()
    }
}

impl Dims for RangeTo<isize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let dim = shape.dim(self.end);
        (0..dim).collect()
    }
}

impl Dims for RangeTo<i32> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let dim = shape.dim(self.end);
        (0..dim).collect()
    }
}

pub trait Shape: Debug {
    fn shape(&self) -> &[usize];

    #[inline]
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    #[inline]
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    #[inline]
    fn shape_at<I: Dim>(&self, i: I) -> usize {
        self.shape()[i.dim_of(self)]
    }

    #[inline]
    fn shape_until<I: Dim>(&self, i: I) -> &[usize] {
        &self.shape()[..=i.dim_of(self)]
    }

    #[inline]
    fn dim<I: Dim>(&self, i: I) -> usize {
        i.dim_of(self)
    }

    #[inline]
    fn dims<D: Dims>(&self, d: D) -> Vec<usize> {
        d.dims_of(self)
    }

    #[inline]
    fn size_of_dims<D: Dims>(&self, d: D) -> usize {
        d.dims_of(self)
            .into_iter()
            .map(|d| self.shape_at(d))
            .product()
    }

    fn shape_transpose(&self) -> Vec<usize> {
        let shape = self.shape();
        let ndim = self.ndim();
        let mut transposed_shape = vec![0; ndim];
        for (i, &s) in shape.iter().enumerate() {
            transposed_shape[ndim - i - 1] = s;
        }
        transposed_shape
    }

    #[inline]
    fn shape_eq<S: Shape + ?Sized>(&self, rhs: &S) -> bool {
        self.shape().eq(rhs.shape())
    }

    #[inline]
    fn shape_ndim_eq<S: Shape + ?Sized>(&self, rhs: &S) -> bool {
        self.ndim() == rhs.ndim()
    }

    #[inline]
    fn shape_size_eq<S: Shape + ?Sized>(&self, rhs: &S) -> bool {
        self.size() == rhs.size()
    }

    fn shape_broadcast<T: Shape + ?Sized>(&self, rhs: &T) -> Result<Vec<usize>> {
        let lhs = self;
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let lhs_ndim = lhs_shape.ndim();
        let rhs_ndim = rhs_shape.ndim();
        let out_ndim = usize::max(lhs_ndim, rhs_ndim);
        let mut out_shape = vec![0; out_ndim];
        for (idx, out_value) in out_shape.iter_mut().enumerate() {
            let rev_idx = out_ndim - idx;
            let l_value = if lhs_ndim < rev_idx {
                1
            } else {
                lhs_shape[lhs_ndim - rev_idx]
            };
            let r_value = if rhs_ndim < rev_idx {
                1
            } else {
                rhs_shape[rhs_ndim - rev_idx]
            };
            *out_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                return Err(Error::IncompatibleShape {
                    lhs: lhs.shape().to_vec(),
                    rhs: rhs.shape().to_vec(),
                });
            }
        }
        Ok(out_shape)
    }

    fn shape_broadcast_matmul<S: Shape + ?Sized>(&self, rhs: &S) -> Result<Vec<usize>> {
        let lhs_in = self;
        let rhs_in = rhs;

        if lhs_in.ndim() == 0 || rhs_in.ndim() == 0 {
            return Err(Error::IncompatibleShape {
                lhs: lhs_in.shape().to_vec(),
                rhs: rhs_in.shape().to_vec(),
            });
        }

        let mut lhs = self.shape().to_vec();
        let mut rhs = rhs.shape().to_vec();

        if lhs.len() == 1 {
            lhs.insert(0, 1);
        }
        if rhs.len() == 1 {
            rhs.push(1);
        }

        let (m, lhs_k) = (lhs[lhs.len() - 2], lhs[lhs.len() - 1]);
        let (rhs_k, n) = (rhs[rhs.len() - 2], rhs[rhs.len() - 1]);

        if lhs_k != rhs_k {
            return Err(Error::IncompatibleShape {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
        }

        if lhs.len() == 2 && rhs.len() == 2 {
            return Ok(vec![m, n]);
        }

        let lhs_b = &lhs[..lhs.len() - 2];
        let rhs_b = &rhs[..rhs.len() - 2];

        let batching = lhs_b.shape_broadcast(rhs_b)?;
        let mut out_shape = [batching.shape(), &[m, n]].concat();

        if lhs_in.ndim() == 1 || rhs_in.ndim() == 1 {
            let erase_start = out_shape.len() - if lhs_in.ndim() == 1 { 2 } else { 1 };
            let erase_end = out_shape.len() - if rhs_in.ndim() == 1 { 0 } else { 1 };
            out_shape.drain(erase_start..erase_end);
        }

        Ok(out_shape)
    }

    fn shape_reduce<T: AsRef<[usize]>>(&self, dims: T, keep_dim: bool) -> Vec<usize> {
        let dims = dims.as_ref();
        let mut out_shape = Vec::new();
        for i in self.dims(..) {
            if !dims.contains(&i) {
                out_shape.push(self.shape_at(i));
            } else if keep_dim {
                out_shape.push(1);
            }
        }
        out_shape
    }
}

impl<'a, T> Shape for &'a T
where
    T: Shape + ?Sized,
{
    fn shape(&self) -> &[usize] {
        (*self).shape()
    }
}

impl Shape for Vec<usize> {
    fn shape(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

impl Shape for [usize] {
    fn shape(&self) -> &[usize] {
        self
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

impl<const N: usize> Shape for [usize; N] {
    fn shape(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}
