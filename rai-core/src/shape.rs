use crate::{Error, Result};
use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

pub trait Dim: Debug {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize;
    fn is_negative(&self) -> bool;
}

impl<'a, T> Dim for &'a T
where
    T: Dim + ?Sized,
{
    fn dim_of<U: Shape + ?Sized>(&self, shape: &U) -> usize {
        (*self).dim_of(shape)
    }

    fn is_negative(&self) -> bool {
        (*self).is_negative()
    }
}

impl Dim for usize {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize {
        assert!(*self < shape.ndim(), "{} < {}", self, shape.ndim());
        *self
    }

    fn is_negative(&self) -> bool {
        false
    }
}

impl Dim for isize {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize {
        let dim = if *self >= 0 {
            *self as usize
        } else {
            self.checked_add_unsigned(shape.ndim()).unwrap() as usize
        };
        assert!(dim < shape.ndim(), "{} < {}", dim, shape.ndim());
        dim
    }

    fn is_negative(&self) -> bool {
        *self < 0
    }
}

impl Dim for i32 {
    fn dim_of<T: Shape + ?Sized>(&self, shape: &T) -> usize {
        let dim = if *self >= 0 {
            *self as usize
        } else {
            self.checked_add_unsigned(shape.ndim() as u32).unwrap() as usize
        };
        assert!(dim < shape.ndim(), "{} < {}", dim, shape.ndim());
        dim
    }

    fn is_negative(&self) -> bool {
        *self < 0
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

impl Dims for RangeToInclusive<usize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let dim = shape.dim(self.end);
        (0..=dim).collect()
    }
}

impl Dims for RangeToInclusive<isize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let dim = shape.dim(self.end);
        (0..=dim).collect()
    }
}

impl Dims for RangeToInclusive<i32> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let dim = shape.dim(self.end);
        (0..=dim).collect()
    }
}

impl Dims for RangeFrom<usize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start);
        let end = shape.ndim();
        if start > end {
            (end + 1..=start).rev().collect()
        } else {
            (start..end).collect()
        }
    }
}

impl Dims for RangeFrom<isize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start);
        let end = shape.ndim();
        if start > end {
            (end + 1..=start).rev().collect()
        } else {
            (start..end).collect()
        }
    }
}

impl Dims for RangeFrom<i32> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start);
        let end = shape.ndim();
        if start > end {
            (end + 1..=start).rev().collect()
        } else {
            (start..end).collect()
        }
    }
}

impl Dims for Range<usize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start);
        let end = shape.dim(self.end);
        if start > end {
            (end + 1..=start).rev().collect()
        } else {
            (start..end).collect()
        }
    }
}

impl Dims for Range<isize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start);
        let end = shape.dim(self.end);
        if start > end {
            (end + 1..=start).rev().collect()
        } else {
            (start..end).collect()
        }
    }
}

impl Dims for Range<i32> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start);
        let end = shape.dim(self.end);
        if start > end {
            (end + 1..=start).rev().collect()
        } else {
            (start..end).collect()
        }
    }
}

impl Dims for RangeInclusive<usize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start());
        let end = shape.dim(self.end());
        if start > end {
            (end..=start).rev().collect()
        } else {
            (start..=end).collect()
        }
    }
}

impl Dims for RangeInclusive<isize> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start());
        let end = shape.dim(self.end());
        if start > end {
            (end..=start).rev().collect()
        } else {
            (start..=end).collect()
        }
    }
}

impl Dims for RangeInclusive<i32> {
    fn dims_of<T: Shape + ?Sized>(&self, shape: &T) -> Vec<usize> {
        let start = shape.dim(self.start());
        let end = shape.dim(self.end());
        if start > end {
            (end..=start).rev().collect()
        } else {
            (start..=end).collect()
        }
    }
}

pub trait Shape: Debug {
    fn shape(&self) -> &[usize];

    #[inline]
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    #[inline]
    fn shape_at<I: Dim>(&self, i: I) -> usize {
        self.shape()[i.dim_of(self)]
    }

    #[inline]
    fn shape_before<const N: usize>(&self) -> [usize; N] {
        self.shape()[..N].try_into().unwrap()
    }

    #[inline]
    fn shape_of<D: Dims>(&self, d: D) -> Vec<usize> {
        let dims = d.dims_of(self);
        dims.into_iter().map(|d| self.shape_at(d)).collect()
    }

    #[inline]
    fn size_of<D: Dims>(&self, d: D) -> usize {
        self.shape_of(d).iter().product()
    }

    #[inline]
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    #[inline]
    fn dim<I: Dim>(&self, i: I) -> usize {
        i.dim_of(self)
    }

    #[inline]
    fn dims<D: Dims>(&self, d: D) -> Vec<usize> {
        d.dims_of(self)
    }

    fn shape_transpose(&self, dim0: usize, dim1: usize) -> Vec<usize> {
        let mut shape = self.shape().to_vec();
        shape[dim0] = self.shape_at(dim1);
        shape[dim1] = self.shape_at(dim0);
        shape
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

    fn shape_expand_left<T: Shape + ?Sized>(&self, rhs: &T) -> Vec<usize> {
        let mut dims = rhs.shape().to_vec();
        dims.extend(self.shape());
        dims
    }

    fn shape_expand_right<T: Shape + ?Sized>(&self, rhs: &T) -> Vec<usize> {
        let mut dims = self.shape().to_vec();
        dims.extend(rhs.shape());
        dims
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

    fn shape_conv<S: Shape + ?Sized>(
        &self,
        kernel: &S,
        padding: &[usize],
        stride: &[usize],
        dilation: &[usize],
        groups: usize,
    ) -> Result<Vec<usize>> {
        let mut out_shape = Vec::with_capacity(self.ndim());
        out_shape.push(self.shape_at(0));
        out_shape.push(kernel.shape_at(0) / groups);
        for i in 2..self.ndim() {
            let s = (self.shape_at(i) + 2 * padding[i - 2]
                - dilation[i - 2] * (kernel.shape_at(i) - 1)
                - 1)
                / stride[i - 2]
                + 1;
            out_shape.push(s);
        }
        Ok(out_shape)
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
