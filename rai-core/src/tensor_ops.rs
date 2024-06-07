use crate::{ops::*, RaiResult, Tensor};

pub trait TensorOps:
    AbsOp
    + AddOp
    + ArgMaxOp
    + ArgMinOp
    + AvgPool1dOp
    + AvgPool2dOp
    + BroadcastOp
    + ConvTranspose1dOp
    + ConvTranspose2dOp
    + Conv1dOp
    + Conv2dOp
    + CosOp
    + DivOp
    + EqOp
    + ErfOp
    + ExpOp
    + FlashAttentionOp
    + FullOp
    + GatherOp
    + GeOp
    + GtOp
    + IndexAddOp
    + IndexSelectOp
    + LeOp
    + LtOp
    + LogSoftmaxOp
    + LogOp
    + Log2Op
    + Log10Op
    + MatMulOp
    + MaxPool1dOp
    + MaxPool2dOp
    + MaximumOp
    + MinimumOp
    + MulOp
    + NarrowOp
    + NegOp
    + NeOp
    + PermuteOp
    + PowerFloatOp
    + RandOp
    + RandnOp
    + ReduceMaxOp
    + ReduceMinOp
    + ReduceSumOp
    + ReshapeOp
    + RsqrtOp
    + ScatterAddOp
    + SignOp
    + SinOp
    + SoftmaxOp
    + SqrtOp
    + SquareOp
    + SubOp
    + TanhOp
    + ToContiguousOp
    + ToDeviceOp
    + ToDTypeOp
    + TransposeOp
    + UpsampleNearest1dOp
    + UpsampleNearest2dOp
{
}

impl TensorOps for Tensor {}
impl<'a> TensorOps for &'a Tensor {}
impl TensorOps for RaiResult<Tensor> {}
impl<'a> TensorOps for RaiResult<&'a Tensor> {}
impl<'a> TensorOps for &'a RaiResult<Tensor> {}
