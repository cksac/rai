use crate::{device, dtype, eval, vjp, AsDType, AsDevice, Func, Shape, Tensor, Value};

fn basis(size: usize, index: usize, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
    let mut data = vec![0u32; size];
    data[index] = 1;
    Tensor::from_array(data, [size], device).to_dtype(dtype)
}

pub fn jacrev<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> Tensor + Clone + 'a
where
    F: Func<K, IN, OUT> + Clone + 'a,
    IN: Value<Tensors = Tensor, Gradient = Tensor>,
    OUT: Value<Tensors = Tensor, Gradient = Tensor>,
{
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
                let i = vin.reshape(y.shape());
                pullback(i).flatten(..)
            })
            .collect::<Vec<_>>();
        Tensor::cat(jv.as_slice(), 0).reshape(in_tensor.shape().shape_expand_right(&y))
    };
    jac_fn
}
