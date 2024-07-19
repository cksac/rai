use crate::{AsDType, AsDevice, Tensor};

pub fn triu(n: usize, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
    let device = device.device();
    let t = Tensor::arange((0u32, n as u32), device);
    let t1 = t.reshape([1, n]).broadcast_to_unchecked([n, n]);
    let t2 = t.reshape([n, 1]).broadcast_to_unchecked([n, n]);
    t1.ge(t2).to_dtype(dtype)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn triu(n: usize, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
        triu(n, dtype, device)
    }
}
