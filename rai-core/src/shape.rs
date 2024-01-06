use crate::{Error, Result};
use std::fmt::Debug;

pub trait Shape: Debug {
    fn dims(&self) -> &[usize];
    fn ndim(&self) -> usize;

    #[inline]
    fn axes(&self) -> Vec<usize> {
        (0..self.ndim()).collect::<Vec<_>>()
    }

    #[inline]
    fn size(&self) -> usize {
        self.dims().iter().product()
    }

    #[inline]
    fn to_vec(&self) -> Vec<usize> {
        self.dims().to_vec()
    }

    fn shape_transpose(&self) -> Vec<usize> {
        let dims = self.dims();
        let ndim = self.ndim();
        let mut transposed_dims = vec![0; ndim];
        for (i, &dim) in dims.iter().enumerate() {
            transposed_dims[ndim - i - 1] = dim;
        }
        transposed_dims
    }
    #[inline]
    fn shape_at(&self, dim: usize) -> usize
    where
        Self: Sized,
    {
        self.dims()[dim]
    }

    #[inline]
    fn shape_eq(&self, rhs: &impl Shape) -> bool
    where
        Self: Sized,
    {
        self.dims().eq(rhs.dims())
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
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx {
                1
            } else {
                lhs_dims[lhs_ndims - rev_idx]
            };
            let r_value = if rhs_ndims < rev_idx {
                1
            } else {
                rhs_dims[rhs_ndims - rev_idx]
            };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                return Err(Error::IncompatibleShape {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                });
            }
        }
        Ok(bcast_dims)
    }

    fn shape_broadcast_matmul(&self, rhs: &impl Shape) -> Result<Vec<usize>>
    where
        Self: Sized,
    {
        let lhs_in = &self;
        let rhs_in = &rhs;

        if lhs_in.ndim() == 0 || rhs_in.ndim() == 0 {
            return Err(Error::IncompatibleShape {
                lhs: lhs_in.to_vec(),
                rhs: rhs_in.to_vec(),
            });
        }

        let mut lhs = self.dims().to_owned();
        let mut rhs = rhs.dims().to_owned();

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
        let mut out_shape = [batching.dims(), &[m, n]].concat();

        if lhs_in.ndim() == 1 || rhs_in.ndim() == 1 {
            let erase_start = out_shape.len() - if lhs_in.ndim() == 1 { 2 } else { 1 };
            let erase_end = out_shape.len() - if rhs_in.ndim() == 1 { 0 } else { 1 };
            out_shape.drain(erase_start..erase_end);
        }

        Ok(out_shape)
    }
}

impl Shape for Vec<usize> {
    fn dims(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

impl Shape for &Vec<usize> {
    fn dims(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

impl Shape for &[usize] {
    fn dims(&self) -> &[usize] {
        self
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

impl<const N: usize> Shape for [usize; N] {
    fn dims(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}

impl<const N: usize> Shape for &[usize; N] {
    fn dims(&self) -> &[usize] {
        self.as_slice()
    }

    fn ndim(&self) -> usize {
        self.len()
    }
}
