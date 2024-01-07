use crate::{Error, Result};
use std::{fmt::Debug, ops::RangeFull};

pub trait DimIndex {
    fn dim_of<T: Shape>(&self, shape: &T) -> usize;
}

impl DimIndex for usize {
    fn dim_of<T: Shape>(&self, shape: &T) -> usize {
        assert!(*self < shape.ndim());
        *self
    }
}

impl DimIndex for RangeFull {
    fn dim_of<T: Shape>(&self, shape: &T) -> usize {
        shape.ndim() - 1
    }
}

impl DimIndex for isize {
    fn dim_of<T: Shape>(&self, shape: &T) -> usize {
        let axis = if *self >= 0 {
            *self as usize
        } else {
            self.checked_add_unsigned(shape.ndim() - 1).unwrap() as usize
        };
        assert!(axis < shape.ndim());
        axis
    }
}

impl DimIndex for i32 {
    fn dim_of<T: Shape>(&self, shape: &T) -> usize {
        let axis = if *self >= 0 {
            *self as usize
        } else {
            self.checked_add_unsigned(shape.ndim() as u32).unwrap() as usize
        };
        assert!(axis < shape.ndim(), "{} < {}", axis, shape.ndim());
        axis
    }
}

pub trait Shape: Debug {
    /// return size of each dimension
    fn shape(&self) -> &[usize];

    /// return number of dimensions
    #[inline]
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// return total element count of the shape
    #[inline]
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// return size of dimension at dim of i
    #[inline]
    fn shape_at<I: DimIndex>(&self, i: I) -> usize
    where
        Self: Sized,
    {
        self.shape()[i.dim_of(self)]
    }

    /// return size of dimensions from 0 to dim of i
    #[inline]
    fn shape_until<I: DimIndex>(&self, i: I) -> &[usize]
    where
        Self: Sized,
    {
        &self.shape()[..=i.dim_of(self)]
    }

    /// return dim index at index i
    #[inline]
    fn dim_at<I: DimIndex>(&self, i: I) -> usize
    where
        Self: Sized,
    {
        i.dim_of(self)
    }

    /// return dim indexes before index i
    #[inline]
    fn dims_until<I: DimIndex>(&self, i: I) -> Vec<usize>
    where
        Self: Sized,
    {
        (0..=i.dim_of(self)).collect()
    }

    /// return dim indexes of the shape
    #[inline]
    fn dims(&self) -> Vec<usize>
    where
        Self: Sized,
    {
        (0..self.ndim()).collect()
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
    fn shape_eq(&self, rhs: &impl Shape) -> bool
    where
        Self: Sized,
    {
        self.shape().eq(rhs.shape())
    }

    #[inline]
    fn shape_ndim_eq(&self, rhs: &impl Shape) -> bool
    where
        Self: Sized,
    {
        self.ndim() == rhs.ndim()
    }

    #[inline]
    fn shape_size_eq(&self, rhs: &impl Shape) -> bool
    where
        Self: Sized,
    {
        self.size() == rhs.size()
    }

    fn shape_broadcast(&self, rhs: &impl Shape) -> Result<Vec<usize>>
    where
        Self: Sized,
    {
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

    fn shape_broadcast_matmul(&self, rhs: &impl Shape) -> Result<Vec<usize>>
    where
        Self: Sized,
    {
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

        let batching = lhs_b.shape_broadcast(&rhs_b)?;
        let mut out_shape = [batching.shape(), &[m, n]].concat();

        if lhs_in.ndim() == 1 || rhs_in.ndim() == 1 {
            let erase_start = out_shape.len() - if lhs_in.ndim() == 1 { 2 } else { 1 };
            let erase_end = out_shape.len() - if rhs_in.ndim() == 1 { 0 } else { 1 };
            out_shape.drain(erase_start..erase_end);
        }

        Ok(out_shape)
    }

    fn shape_reduce<T: AsRef<[usize]>>(&self, dims: T, keep_dim: bool) -> Vec<usize>
    where
        Self: Sized,
    {
        let dims = dims.as_ref();
        let mut out_shape = Vec::new();
        for i in self.dims() {
            if !dims.contains(&i) {
                out_shape.push(self.shape_at(i));
            } else if keep_dim {
                out_shape.push(1);
            }
        }
        out_shape
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

impl Shape for &Vec<usize> {
    fn shape(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

impl Shape for &[usize] {
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

impl<const N: usize> Shape for &[usize; N] {
    fn shape(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

#[test]
fn test_shape() {
    let s = [1, 2, 3, 4];
    let d = s.dims_until(-1);
    dbg!(s, d);

    let s = [1];
    let d = s.dims_until(-1);
    dbg!(s, d);

    let s = [];
    let d = s.dims_until(-1);
    dbg!(s, d);
}
