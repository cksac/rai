use crate::{Dim, Shape, Tensor};

#[track_caller]
pub fn chunk(x: &Tensor, chunks: usize, dim: impl Dim) -> Vec<Tensor> {
    let dim = x.dim(dim);
    let size = x.size(dim);
    if size < chunks {
        (0..size).map(|i| x.narrow(dim, i, 1)).collect::<Vec<_>>()
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
            let tensor = x.narrow(dim, sum_chunk_size, chunk_size);
            tensors.push(tensor);
            sum_chunk_size += chunk_size
        }
        tensors
    }
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn chunk(&self, chunks: usize, dim: impl Dim) -> Vec<Tensor> {
        chunk(self, chunks, dim)
    }
}
