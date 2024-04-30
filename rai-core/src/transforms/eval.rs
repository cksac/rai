use crate::{dispatch::eval_rule, Tensor, TensorIter};
use std::collections::BTreeSet;

pub trait EvalArgs {
    fn outputs(&self) -> impl Iterator<Item = &Tensor>;
    fn retain_graph(&self) -> bool {
        false
    }
}

impl<T> EvalArgs for T
where
    T: TensorIter,
{
    fn outputs(&self) -> impl Iterator<Item = &Tensor> {
        self.tensor_iter()
    }
}

impl<T> EvalArgs for (T, bool)
where
    T: TensorIter,
{
    fn outputs(&self) -> impl Iterator<Item = &Tensor> {
        self.0.tensor_iter()
    }

    fn retain_graph(&self) -> bool {
        self.1
    }
}
pub fn eval<T: EvalArgs>(args: T) {
    let mut tape = BTreeSet::new();
    let mut stack = Vec::new();

    // use iterative instead of recursive to avoid stack overflow
    // TODO: use proper topo sort algorithm, now sort by id in BTreeSet
    for output in args.outputs() {
        stack.push(output.clone());
    }

    while let Some(t) = stack.pop() {
        if t.is_evaluated() || tape.contains(&t) {
            continue;
        }
        tape.insert(t.clone());
        for input in t.inputs().iter() {
            stack.push(input.clone());
        }
    }
    // Process the sorted tensors in the tape
    for t in tape.into_iter() {
        {
            let device = t.device().clone_boxed();
            let device = device.as_ref();
            let primitive = t.primitive().clone_boxed();
            let primitive = primitive.as_ref();
            let inputs = &*t.inputs();
            let rule = eval_rule(device, primitive).unwrap_or_else(|| {
                panic!(
                    "no eval rule for device: {:?}, primitive: {:?}",
                    device, primitive
                )
            });
            rule.eval(device, primitive, inputs, &t);
        }
        if !args.retain_graph() {
            t.detach();
        }
    }
}
