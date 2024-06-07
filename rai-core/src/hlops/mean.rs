use crate::{ops::ReduceArgs, RaiResult, Shape, Tensor, TryAsTensor};

#[track_caller]
pub fn mean<T: ReduceArgs>(x: impl TryAsTensor, args: T) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let elem_count = x.dims_elem_count(args.dims()) as f64;
    x.sum(args) / elem_count
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn mean<T: ReduceArgs>(&self, args: T) -> RaiResult<Tensor> {
        mean(self, args)
    }
}
