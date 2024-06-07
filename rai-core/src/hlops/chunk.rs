use crate::{Dim, RaiResult, Shape, Tensor, TryAsTensor};

#[track_caller]
pub fn chunk(x: impl TryAsTensor, chunks: usize, dim: impl Dim) -> RaiResult<Vec<Tensor>> {
    let x = crate::try_get! { x.try_as_tensor() };
    let dim = x.dim(dim);
    let size = x.size(dim);
    if size < chunks {
        (0..size).map(|i| x.narrow(dim, i, 1)).collect()
    } else {
        let chunk_size = size / chunks;
        let cnt_additional = size % chunks;
        let mut tensors = vec![];
        let mut sum_chunk_size = 0;
        for i in 0..chunks {
            let chunk_size = if i < cnt_additional {
                chunk_size + 1
            } else {
                chunk_size
            };
            let tensor = crate::try_get! { x.narrow(dim, sum_chunk_size, chunk_size) };
            tensors.push(tensor);
            sum_chunk_size += chunk_size
        }
        RaiResult::from_val(tensors)
    }
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn chunk(&self, chunks: usize, dim: impl Dim) -> RaiResult<Vec<Tensor>> {
        chunk(self, chunks, dim)
    }
}
