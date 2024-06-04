use crate::{dim::Before, Dim, Dims, Error, Result};
use std::{cmp::Ordering, fmt::Debug};

pub trait Shape: Debug {
    fn shape(&self) -> &[usize];

    #[inline]
    fn rank(&self) -> usize {
        self.shape().len()
    }

    #[inline]
    fn size<D: Dim>(&self, dim: D) -> usize {
        dim.size_of(self)
    }

    #[inline]
    fn size_unchecked<D: Dim>(&self, dim: D) -> usize {
        dim.size_of_unchecked(self)
    }

    #[inline]
    fn sizes<O, D: Dims<O>>(&self, dims: D) -> O {
        dims.sizes_of(self)
    }

    #[inline]
    fn sizes_unchecked<O, D: Dims<O>>(&self, dims: D) -> O {
        dims.sizes_of_unchecked(self)
    }

    #[inline]
    fn dim<D: Dim>(&self, dim: D) -> usize {
        dim.dim_of(self)
    }

    #[inline]
    fn dim_unchecked<D: Dim>(&self, dim: D) -> usize {
        dim.dim_of_unchecked(self)
    }

    #[inline]
    fn dims<O, D: Dims<O>>(&self, dims: D) -> O {
        dims.dims_of(self)
    }

    #[inline]
    fn dims_unchecked<O, D: Dims<O>>(&self, dims: D) -> O {
        dims.dims_of_unchecked(self)
    }

    #[inline]
    fn elem_count(&self) -> usize {
        self.shape().iter().product()
    }

    #[inline]
    fn dims_elem_count<O: Shape, D: Dims<O>>(&self, dims: D) -> usize {
        let s = dims.sizes_of(self);
        s.elem_count()
    }

    #[inline]
    fn dims_elem_count_unchecked<O: Shape, D: Dims<O>>(&self, dims: D) -> usize {
        let s = dims.sizes_of_unchecked(self);
        s.elem_count()
    }

    fn shape_transpose(&self, dim0: usize, dim1: usize) -> Vec<usize> {
        let mut shape = self.shape().to_vec();
        shape[dim0] = self.size(dim1);
        shape[dim1] = self.size(dim0);
        shape
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
            match lhs.rank().cmp(&rhs.rank()) {
                Ordering::Less => {
                    let mut lhs_shape = vec![1; rhs.rank() - lhs.rank()];
                    lhs_shape.extend(lhs.shape());
                    lhs_b = true;
                    (lhs_shape, rhs.shape().to_vec())
                }
                Ordering::Greater => {
                    let mut rhs_shape = vec![1; lhs.rank() - rhs.rank()];
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
        if lhs.rank() < 2 || rhs.rank() < 2 {
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
                out_shape.push(self.size_unchecked(i));
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
        let [b_size, c_in, l_in] = self.sizes(Before::<3>);
        let [c_out, c_in_k, k_size] = kernel.sizes(Before::<3>);
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
        let [b_size, c_in, h_in, w_in] = self.sizes(Before::<4>);
        let [c_out, c_in_k, h_k, w_k] = kernel.sizes(Before::<4>);
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
        let [b_size, c_in, l_in] = self.sizes(Before::<3>);
        let [c_in_k, c_out, k_siz] = kernel.sizes(Before::<3>);
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
        let [b_size, c_in, h_in, w_in] = self.sizes(Before::<4>);
        let [c_in_k, c_out, h_k, w_k] = kernel.sizes(Before::<4>);
        assert_eq!(c_in, c_in_k);
        let h_out = (h_in - 1) * stride[0] + dilation[0] * (h_k - 1) + output_padding[0] + 1
            - 2 * padding[0];
        let w_out = (w_in - 1) * stride[1] + dilation[1] * (w_k - 1) + output_padding[1] + 1
            - 2 * padding[1];
        vec![b_size, c_out, h_out, w_out]
    }

    fn shape_avg_pool1d(&self, kernel_size: usize, stride: usize, padding: usize) -> Vec<usize> {
        let [b_size, c_in, l_in] = self.sizes(Before::<3>);
        let l_out = (l_in + 2 * padding - kernel_size) / stride + 1;
        vec![b_size, c_in, l_out]
    }

    fn shape_avg_pool2d(
        &self,
        kernel_size: &(usize, usize),
        stride: &(usize, usize),
        padding: &(usize, usize),
    ) -> Vec<usize> {
        let [b_size, c_in, h_in, w_in] = self.sizes(Before::<4>);
        let h_out = (h_in + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
        vec![b_size, c_in, h_out, w_out]
    }

    fn shape_max_pool1d(
        &self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dialation: usize,
    ) -> Vec<usize> {
        let [b_size, c_in, l_in] = self.sizes(Before::<3>);
        let l_out = (l_in + 2 * padding - dialation * (kernel_size - 1) - 1) / stride + 1;
        vec![b_size, c_in, l_out]
    }

    fn shape_max_pool2d(
        &self,
        kernel_size: &(usize, usize),
        stride: &(usize, usize),
        padding: &(usize, usize),
        dialation: &(usize, usize),
    ) -> Vec<usize> {
        let [b_size, c_in, h_in, w_in] = self.sizes(Before::<4>);
        let h_out = (h_in + 2 * padding.0 - dialation.1 * (kernel_size.0 - 1) - 1) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - dialation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1;
        vec![b_size, c_in, h_out, w_out]
    }

    fn shape_upsample_nearest1d(&self, size: usize) -> Vec<usize> {
        let [b_size, c_in, _l_in] = self.sizes(Before::<3>);
        vec![b_size, c_in, size]
    }

    fn shape_upsample_nearest2d(&self, size: &(usize, usize)) -> Vec<usize> {
        let [b_size, c_in, _h_in, _w_in] = self.sizes(Before::<4>);
        vec![b_size, c_in, size.0, size.1]
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

    fn rank(&self) -> usize {
        self.len()
    }
}

impl Shape for [usize] {
    fn shape(&self) -> &[usize] {
        self
    }

    fn rank(&self) -> usize {
        self.len()
    }
}

impl<const N: usize> Shape for [usize; N] {
    fn shape(&self) -> &[usize] {
        self.as_slice()
    }

    fn rank(&self) -> usize {
        self.len()
    }
}
