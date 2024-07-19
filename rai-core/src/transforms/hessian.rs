use crate::{jacfwd, jacrev, Func, Tensor, Value};

pub fn hessian<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> Tensor + Clone + 'a
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value<Tensors = Tensor, Gradient = Tensor>,
    OUT: Value<Tensors = Tensor, Gradient = Tensor>,
{
    jacfwd(jacrev(func))
}
