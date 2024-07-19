use crate::{vjp, Func, Shape, Tensor, Value};

pub fn jacrev<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> Tensor + Clone + 'a
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value<Tensors = Tensor, Gradient = Tensor>,
    OUT: Value<Tensors = Tensor, Gradient = Tensor>,
{
    // TODO: use vmap later
    let jac_fn = move |input: IN| {
        let in_tensor = input.tensors();
        let func = func.clone();
        let (y, pullback) = vjp(func, input);
        let y = y.tensors();
        let y_flat = y.flatten(..);
        let l = y_flat.shape().size(0);
        let y_eye = Tensor::eye(l, y.dtype(), y.device());
        let jv = y_eye
            .chunk(l, 0)
            .into_iter()
            .map(|vin| {
                let i = vin.reshape(&y);
                pullback(i).flatten(..)
            })
            .collect::<Vec<_>>();
        Tensor::cat(jv.as_slice(), 0).reshape(in_tensor.shape().shape_expand_right(&y))
    };
    jac_fn
}
