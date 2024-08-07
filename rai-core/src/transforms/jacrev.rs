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
        let (out, pullback) = vjp(func, input);
        let out_tensor = out.tensors();
        let out_tensor_flat = out_tensor.flatten(..);
        let l = out_tensor_flat.shape().size(0);
        let out_eye = Tensor::eye(l, out_tensor.dtype(), out_tensor.device());
        let jv = out_eye
            .chunk(l, 0)
            .into_iter()
            .map(|vin| {
                let i = vin.reshape(&out_tensor);
                pullback(i).flatten(..)
            })
            .collect::<Vec<_>>();
        Tensor::cat(jv.as_slice(), 0).reshape(in_tensor.shape().shape_expand_right(&out_tensor))
    };
    jac_fn
}
