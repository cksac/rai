use crate::ops::ReduceArgs;
use crate::{Dims, RaiResult, Shape, Tensor, TryAsTensor};

pub trait VarArgs: ReduceArgs {
    fn ddof(&self) -> usize {
        0
    }
}

impl<T> VarArgs for T where T: Dims<Vec<usize>> {}

impl<T> VarArgs for (T, bool) where T: Dims<Vec<usize>> {}

impl<T> VarArgs for (T, usize)
where
    T: Dims<Vec<usize>>,
{
    fn ddof(&self) -> usize {
        self.1
    }
}

impl<T> VarArgs for (T, bool, usize)
where
    T: Dims<Vec<usize>>,
{
    fn ddof(&self) -> usize {
        self.2
    }
}

#[track_caller]
pub fn var<T: VarArgs>(x: impl TryAsTensor, args: T) -> RaiResult<Tensor> {
    let x = crate::try_get! { x.try_as_tensor() };
    let elem_count = x.dims_elem_count(args.dims());
    let m = x.mean((args.dims(), args.keep_dim()));
    let s = (x - m).square().sum((args.dims(), args.keep_dim()));
    s / (elem_count - args.ddof()) as f32
}

crate::impl_op! {
    #[inline]
    #[track_caller]
    pub fn var<T: VarArgs>(&self, args: T) -> RaiResult<Tensor> {
        var(self, args)
    }
}
