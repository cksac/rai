use crate::{
    primitives::{Full, ToDType, ToDevice},
    Cpu, Cuda, Func, Metal, Shape, Tensor, TensorIter, Value, F32, F64,
};
use std::{
    any::{self, TypeId},
    cell::RefCell,
    collections::{BTreeSet, HashMap},
    panic::Location,
};

fn is_full(t: &Tensor) -> Option<String> {
    if let Some(f) = t.primitive().as_any().downcast_ref::<Full<F32>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(f) = t.primitive().as_any().downcast_ref::<Full<F64>>() {
        Some(format!("{:?}", f.val))
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<F32>>() {
        is_full(t.inputs().first().unwrap())
    } else if let Some(_) = t.primitive().as_any().downcast_ref::<ToDType<F64>>() {
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

// #[derive(Debug, Default)]
// struct OptimizeCache {
//     cache: HashMap<String, CacheEntry>,
// }
// impl OptimizeCache {
//     fn get(&self, key: &str) -> Option<&CacheEntry> {
//         self.cache.get(key)
//     }

//     fn insert(&mut self, key: String, entry: CacheEntry) {
//         self.cache.insert(key, entry);
//     }
// }

// #[derive(Debug, Default, Clone)]
// struct CacheEntry {
//     inputs: Vec<Tensor>,
//     outputs: Vec<Tensor>,
//     tape: BTreeSet<Tensor>,
// }

// thread_local! {
//     static  OPTIMIZE_CACHE: RefCell<OptimizeCache> = Default::default();
// }

#[track_caller]
pub fn optimize<'a, K, IN, OUT, F>(func: F) -> impl Fn(IN) -> OUT + 'a
where
    F: Func<K, IN, OUT> + 'a,
    IN: Value,
    OUT: Value,
{
    //let fun_key = format!("{}@{}", any::type_name_of_val(&func), Location::caller());
    move |input| {
        //let input_tensors = input.to_tensor_vec();
        // stil need to constuct graph before optimize, can't use cached graph? due to OUT may contain non-tensor outputs...
        let output = func.invoke(input);
        //let output_tensors = output.to_tensor_vec();
        // let cached = OPTIMIZE_CACHE.with(|cache| cache.borrow().get(&fun_key).cloned());
        // if let Some(g) = cached {
        //     let inputs = g.inputs;
        //     let outputs = g.outputs;
        //     let real_inputs = inputs
        //         .iter()
        //         .zip(input_tensors)
        //         .map(|(a, b)| (a.id(), b))
        //         .collect::<HashMap<_, _>>();
        //     let real_outputs = outputs
        //         .iter()
        //         .zip(output_tensors.iter())
        //         .map(|(a, b)| (a.id(), b))
        //         .collect::<HashMap<_, _>>();

        //     let tape = g.tape;
        //     for t in tape.iter() {
        //         if !t.inputs().is_empty() {
        //             // replace real inputs
        //             let inputs = t
        //                 .inputs()
        //                 .iter()
        //                 .map(|x| {
        //                     if let Some(v) = real_inputs.get(&x.id()).cloned() {
        //                         v
        //                     } else {
        //                         x.clone()
        //                     }
        //                 })
        //                 .collect();
        //             t.set_inputs(inputs);
        //         }
        //         if let Some(r) = real_outputs.get(&t.id()) {
        //             dbg!(&r, &t);
        //             // TODO: should replace primitive also
        //             r.set_inputs(t.inputs().iter().cloned().collect());
        //         }
        //     }
        //     return output;
        // }

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

        // let entry = CacheEntry {
        //     inputs: input_tensors,
        //     outputs: output_tensors,
        //     tape,
        // };
        // OPTIMIZE_CACHE.with(|cache| cache.borrow_mut().insert(fun_key.clone(), entry));
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
