use std::any::{Any, TypeId};

use crate::{
    dispatch::{Dispatch, Eval},
    Backend, Primitive, Shape, Tensor,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RaiExpr;

pub struct AbstractTensor {}

type Data = AbstractTensor;

impl Backend for RaiExpr {
    fn clone_boxed(&self) -> Box<dyn Backend> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn data_type_id(&self) -> std::any::TypeId {
        TypeId::of::<Data>()
    }

    fn equal(&self, rhs: &dyn Backend) -> bool {
        rhs.as_any()
            .downcast_ref()
            .map_or(false, |other| self == other)
    }
}

impl<P> Eval<RaiExpr, P> for Dispatch<RaiExpr, P>
where
    P: Primitive + Clone + Send + Sync + 'static,
{
    fn eval(&self, _: &RaiExpr, primitive: &P, inputs: &[Tensor], output: &Tensor) {
        let out_id = output.id();
        let out_shape = output.shape();
        let out_ty = format!("{:?}", output.dtype()).to_lowercase();
        let op = primitive.dot_label().to_lowercase();
        let inputs = inputs
            .iter()
            .map(|v| format!("%{}", v.id()))
            .collect::<Vec<_>>()
            .join(" ");
        println!("%{out_id}:{out_ty}{out_shape:?} = {op} {inputs}");
    }
}
