use crate::{
    primitives::{Full, ToDType, ToDevice},
    Cpu, Cuda, Func, Metal, Shape, Tensor, TensorIter, Value, BF16, F16, F32, F64, I64, U32, U8,
};
use std::{
    any::TypeId,
    collections::{BTreeSet, HashMap},
};

fn is_full(t: &Tensor) -> Option<String> {
    if let Some(f) = t.primitive().as_any().downcast_ref::<Full<F32>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(f) = t.primitive().as_any().downcast_ref::<Full<F64>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(f) = t.primitive().as_any().downcast_ref::<Full<BF16>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(f) = t.primitive().as_any().downcast_ref::<Full<F16>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(f) = t.primitive().as_any().downcast_ref::<Full<U32>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(f) = t.primitive().as_any().downcast_ref::<Full<U8>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(f) = t.primitive().as_any().downcast_ref::<Full<I64>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<F32>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<F64>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<BF16>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<F16>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<U32>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<U8>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<I64>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDevice<Cpu>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDevice<Cuda>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDevice<Metal>>() {
        is_full(t.inputs().first().unwrap())
    } else {
        None
    }
}

// TODO: optimize cache
pub fn optimize<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> OUT + 'a
where
    F: Func<K, IN, OUT> + 'a,
    IN: Value,
    OUT: Value,
{
    move |input| {
        let output = func.invoke(input);

        let mut tape = BTreeSet::new();
        let mut stack = Vec::new();

        // use iterative instead of recursive to avoid stack overflow
        // TODO: use proper topo sort algorithm, now sort by id in BTreeSet
        for output in output.tensors().tensor_iter() {
            stack.push(output.clone());
        }

        while let Some(t) = stack.pop() {
            if tape.contains(&t) {
                continue;
            }
            tape.insert(t.clone());
            for input in t.inputs().iter() {
                stack.push(input.clone());
            }
        }

        // (device, dtype, val:shape) -> Tensor
        let mut constants = HashMap::<(TypeId, TypeId, String), Tensor>::new();
        for t in tape.iter() {
            if let Some(val) = is_full(&t) {
                let key = (
                    t.device().as_any().type_id(),
                    t.dtype().as_any().type_id(),
                    format!("{:?}{:?}", val, t.shape()),
                );
                if constants.get(&key).is_none() {
                    constants.insert(key, t.clone());
                }
            }

            if !t.inputs().is_empty() {
                // constant literal sharing
                let inputs = t
                    .inputs()
                    .iter()
                    .map(|x| {
                        if let Some(v) = is_full(x) {
                            let key = (
                                x.device().as_any().type_id(),
                                x.dtype().as_any().type_id(),
                                format!("{:?}{:?}", v, x.shape()),
                            );
                            constants.get(&key).unwrap().clone()
                        } else {
                            x.clone()
                        }
                    })
                    .collect();
                t.set_inputs(inputs);
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use crate::{Cpu, Tensor, F32};

    fn loss_fn(x: &Tensor, y: &Tensor) -> Tensor {
        x * 2.0 * 2.0 + y
    }

    #[test]
    fn test_optimize() {
        let f = |x: &Tensor, y: &Tensor| x * 2.0 * 2.0 + y;
        let g = super::optimize(f);
        let x = Tensor::rand([], F32, Cpu);
        let y = Tensor::rand([], F32, Cpu);
        let out = g((&x, &y));
        println!("{}", out.dot_graph());

        let x = Tensor::rand([], F32, Cpu);
        let y = Tensor::rand([], F32, Cpu);
        let out = g((&x, &y));
        println!("{}", out.dot_graph());

        let g = super::optimize(loss_fn);
        let x = Tensor::rand([], F32, Cpu);
        let y = Tensor::rand([], F32, Cpu);
        let out = g((&x, &y));
        println!("{}", out.dot_graph());

        let x = Tensor::rand([], F32, Cpu);
        let y = Tensor::rand([], F32, Cpu);
        let out = g((&x, &y));
        println!("{}", out.dot_graph());
    }
}
