use crate::{linearize, Func, Shape, Tensor, Value};

pub fn jacfwd<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> Tensor + Clone + 'a
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value<Tensors = Tensor, Gradient = Tensor>,
    OUT: Value<Tensors = Tensor, Gradient = Tensor>,
{
    // TODO: use vmap later
    let jac_fn = move |input: IN| {
        let in_tensor = input.tensors();
        let func = func.clone();
        let (out, pushforward) = linearize(func, input);
        let out_tensor = out.tensors();
        let in_tensor_flat = in_tensor.flatten(..);
        let l = in_tensor_flat.shape().size(0);
        let in_eye = Tensor::eye(l, in_tensor_flat.dtype(), in_tensor_flat.device());
        let jv = in_eye
            .chunk(l, 0)
            .into_iter()
            .map(|vin| {
                let i = vin.reshape(&in_tensor);
                pushforward(i).flatten(..)
            })
            .collect::<Vec<_>>();
        Tensor::cat(jv.as_slice(), 0).reshape(in_tensor.shape().shape_expand_right(&out_tensor))
    };
    jac_fn
}
