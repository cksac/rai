use crate::{AsDType, AsDevice, Error, Op, Shape, Tensor};

#[derive(Clone, Debug)]
pub struct Err {
    op: Box<dyn Op>,
    err: Error,
}

impl Err {
    pub fn new(op: Box<dyn Op>, err: Error) -> Self {
        Err { op, err }
    }
}

impl Op for Err {
    fn clone_boxed(&self) -> Box<dyn Op> {
        Box::new(Err {
            op: self.op.clone_boxed(),
            err: self.err.clone(),
        })
    }

    fn dot_label(&self) -> String {
        format!("OpErr({}, {:?})", self.op.dot_label(), self.err)
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
        _primals: &[crate::Tensor],
        _cotangent: &crate::Tensor,
    ) -> Vec<crate::Tensor> {
        vec![]
    }

    fn err(&self) -> Option<&Error> {
        Some(&self.err)
    }
}

pub fn err(
    device: impl AsDevice,
    dtype: impl AsDType,
    shape: impl Shape,
    op: impl Into<Box<dyn Op>>,
    inputs: impl Into<Vec<Tensor>>,
    e: Error,
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
        e: Error,
    ) -> Tensor {
        err(device, dtype, shape, op, inputs, e)
    }
}
