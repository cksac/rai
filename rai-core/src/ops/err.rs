use std::borrow::Cow;

use crate::{AsDType, AsDevice, Op, OpError, Shape, Tensor};

#[derive(Clone, Debug)]
pub struct Err {
    op: Box<dyn Op>,
    err: OpError,
}

impl Err {
    pub fn new(op: Box<dyn Op>, err: OpError) -> Self {
        Err { op, err }
    }
}

impl Op for Err {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("Err")
    }

    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(Err {
            op: self.op.clone_boxed(),
            err: self.err.clone(),
        })
    }

    fn dot_label(&self) -> String {
        format!(
            "{}\\nERROR: {}",
            self.op.dot_label(),
            format!("{:?}", self.err)
                .replace("{ ", "(")
                .replace(" }", ")")
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn jvp(
        &self,
        output: &crate::Tensor,
        _primals: &[crate::Tensor],
        _tangents: &[crate::Tensor],
    ) -> crate::Tensor {
        output.zeros_like()
    }

    fn vjp(
        &self,
        _output: &crate::Tensor,
        primals: &[crate::Tensor],
        _cotangent: &crate::Tensor,
    ) -> Vec<crate::Tensor> {
        primals.iter().map(|x| x.zeros_like()).collect()
    }

    fn err(&self) -> Option<&OpError> {
        Some(&self.err)
    }
}

pub fn err(
    device: impl AsDevice,
    dtype: impl AsDType,
    shape: impl Shape,
    op: impl Into<Box<dyn Op>>,
    inputs: impl Into<Vec<Tensor>>,
    e: OpError,
) -> Tensor {
    let op = Err::new(op.into(), e);
    Tensor::new(device, dtype, shape, op, inputs)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn err(
        device: impl AsDevice,
        dtype: impl AsDType,
        shape: impl Shape,
        op: impl Into<Box<dyn Op>>,
        inputs: impl Into<Vec<Tensor>>,
        e: OpError,
    ) -> Tensor {
        err(device, dtype, shape, op, inputs, e)
    }
}
