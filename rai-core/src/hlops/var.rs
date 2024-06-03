use crate::ops::ReduceArgs;
use crate::{Dims, Shape, Tensor};

pub trait VarArgs: ReduceArgs {
    fn ddof(&self) -> usize {
        0
    }
}

impl<T> VarArgs for T where T: Dims {}

impl<T> VarArgs for (T, bool) where T: Dims {}

impl<T> VarArgs for (T, usize)
where
    T: Dims,
{
    fn ddof(&self) -> usize {
        self.1
    }
}

impl<T> VarArgs for (T, bool, usize)
where
    T: Dims,
{
    fn ddof(&self) -> usize {
        self.2
    }
}

#[track_caller]
pub fn var<T: VarArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of(args.dims());
    let m = x.mean((args.dims(), args.keep_dim()));
    let s = (x - m).square().sum((args.dims(), args.keep_dim()));
    s / (elem_count - args.ddof()) as f32
}
