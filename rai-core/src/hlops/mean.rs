use crate::{ops::ReduceArgs, Shape, Tensor};

#[track_caller]
pub fn mean<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of(args.dims()) as f64;
    x.sum(args) / elem_count
}
