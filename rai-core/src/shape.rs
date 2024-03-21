use crate::{Error, Result};
use std::{
    cmp::Ordering,
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

    fn shape_broadcast_to<T: Shape + ?Sized>(&self, rhs: &T) -> Result<(Vec<usize>, bool, bool)> {
        let lhs = self;
        let mut lhs_b = false;
        let mut rhs_b = false;
        let (lhs_shape, rhs_shape) = {
            match lhs.ndim().cmp(&rhs.ndim()) {
                Ordering::Less => {
                    let mut lhs_shape = vec![1; rhs.ndim() - lhs.ndim()];
                    lhs_shape.extend(lhs.shape());
                    lhs_b = true;
                    (lhs_shape, rhs.shape().to_vec())
                }
                Ordering::Greater => {
                    let mut rhs_shape = vec![1; lhs.ndim() - rhs.ndim()];
                    rhs_shape.extend(rhs.shape());
                    rhs_b = true;
                    (lhs.shape().to_vec(), rhs_shape)
                }
                Ordering::Equal => (lhs.shape().to_vec(), rhs.shape().to_vec()),
            }
        };
        let mut out_shape = Vec::with_capacity(lhs_shape.len());
        for (l, r) in lhs_shape.into_iter().zip(rhs_shape.into_iter()) {
            if l == r {
                out_shape.push(l);
            } else if l == 1 {
                out_shape.push(r);
                lhs_b = true;
            } else if r == 1 {
                out_shape.push(l);
                rhs_b = true;
            } else {
                return Err(Error::IncompatibleShape {
                    lhs: lhs.shape().to_vec(),
                    rhs: rhs.shape().to_vec(),
                });
            }
        }
        Ok((out_shape, lhs_b, rhs_b))
    }

    #[allow(clippy::type_complexity)]
    fn shape_broadcast_matmul<S: Shape + ?Sized>(
        &self,
        rhs: &S,
    ) -> Result<(Vec<usize>, Vec<usize>, Vec<usize>, bool, bool)> {
        let lhs = self.shape();
        let rhs = rhs.shape();
        if lhs.ndim() < 2 || rhs.ndim() < 2 {
            return Err(Error::IncompatibleShape {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
            });
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
            return Ok((vec![m, n], lhs.to_vec(), rhs.to_vec(), false, false));
        }
        let lhs_bs = &lhs[..lhs.len() - 2];
        let rhs_bs = &rhs[..rhs.len() - 2];
        let (batching, lhs_b, rhs_b) = lhs_bs.shape_broadcast_to(rhs_bs)?;
        let out_shape = [batching.shape(), &[m, n]].concat();
        let lhs = [batching.shape(), &[m, lhs_k]].concat();
        let rhs = [batching.shape(), &[rhs_k, n]].concat();
        Ok((out_shape, lhs, rhs, lhs_b, rhs_b))
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

    fn shape_conv1d<S: Shape + ?Sized>(
        &self,
        kernel: &S,
        padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Vec<usize> {
        let [b_size, c_in, l_in] = self.shape_before::<3>();
        let [c_out, c_in_k, k_size] = kernel.shape_before::<3>();
        assert_eq!(c_in, c_in_k);
        let l_out = (l_in + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
        vec![b_size, c_out, l_out]
    }

    fn shape_conv2d<S: Shape + ?Sized>(
        &self,
        kernel: &S,
        padding: &[usize; 2],
        stride: &[usize; 2],
        dilation: &[usize; 2],
    ) -> Vec<usize> {
        let [b_size, c_in, h_in, w_in] = self.shape_before::<4>();
        let [c_out, c_in_k, h_k, w_k] = kernel.shape_before::<4>();
        assert_eq!(c_in, c_in_k);
        let h_out = (h_in + 2 * padding[0] - dilation[0] * (h_k - 1) - 1) / stride[0] + 1;
        let w_out = (w_in + 2 * padding[1] - dilation[1] * (w_k - 1) - 1) / stride[1] + 1;
        vec![b_size, c_out, h_out, w_out]
    }

    fn shape_conv_transpose1d<S: Shape + ?Sized>(
        &self,
        kernel: &S,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
    ) -> Vec<usize> {
        let [b_size, c_in, l_in] = self.shape_before::<3>();
        let [c_in_k, c_out, k_siz] = kernel.shape_before::<3>();
        assert_eq!(c_in, c_in_k);
        let l_out = (l_in - 1) * stride + dilation * (k_siz - 1) + output_padding + 1 - 2 * padding;
        vec![b_size, c_out, l_out]
    }

    fn shape_conv_transpose2d<S: Shape + ?Sized>(
        &self,
        kernel: &S,
        padding: &[usize; 2],
        output_padding: &[usize; 2],
        stride: &[usize; 2],
        dilation: &[usize; 2],
    ) -> Vec<usize> {
        let [b_size, c_in, h_in, w_in] = self.shape_before::<4>();
        let [c_in_k, c_out, h_k, w_k] = kernel.shape_before::<4>();
        assert_eq!(c_in, c_in_k);
        let h_out = (h_in - 1) * stride[0] + dilation[0] * (h_k - 1) + output_padding[0] + 1
            - 2 * padding[0];
        let w_out = (w_in - 1) * stride[1] + dilation[1] * (w_k - 1) + output_padding[1] + 1
            - 2 * padding[1];
        vec![b_size, c_out, h_out, w_out]
    }

    fn shape_max_pool2d(
        &self,
        kernel_size: &(usize, usize),
        stride: &(usize, usize),
        padding: &(usize, usize),
        dialation: &(usize, usize),
    ) -> Vec<usize> {
        let [b_size, c_in, h_in, w_in] = self.shape_before::<4>();
        let h_out = (h_in + 2 * padding.0 - dialation.1 * (kernel_size.0 - 1) - 1) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - dialation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1;
        vec![b_size, c_in, h_out, w_out]
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
