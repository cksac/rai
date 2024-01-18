use rai_core::Tensor;

macro_rules! impl_activation {
    ($M:ty, $OP:tt) => {
        impl rai_core::ValuAssociated for $M {
            type ValueType = rai_core::ModuleType;
            type Tensors = ();
            type Gradient = ();
        }

        impl rai_core::Module for $M {
            type Input<'i> = &'i Tensor;
            type Output<'o> = Tensor;
            fn forward<'i, 'o>(&self, x: Self::Input<'i>) -> Self::Output<'o> {
                rai_core::ops::$OP(x)
            }
        }

        impl rai_core::NonTrainableModule for $M {}
    };
}

#[derive(Clone, Debug, Copy)]
pub struct Relu;
impl_activation!(Relu, relu);

// TODO: GELU
#[derive(Clone, Debug, Copy)]
pub struct Gelu;
impl_activation!(Gelu, relu);
