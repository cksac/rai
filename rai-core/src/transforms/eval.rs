use std::collections::BTreeSet;

use crate::{dispatch::eval_rule, utils::topological_sort, Backend, Tensor, TensorIter};

pub trait EvalArgs {
    fn outputs(&self) -> impl Iterator<Item = &Tensor>;
    fn retain_graph(&self) -> bool {
        false
    }
    fn backend(&self) -> Option<Box<dyn Backend>> {
        None
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

impl<T, B> EvalArgs for (T, bool, B)
where
    T: TensorIter,
    B: Backend,
{
    fn outputs(&self) -> impl Iterator<Item = &Tensor> {
        self.0.tensor_iter()
    }

    fn retain_graph(&self) -> bool {
        self.1
    }

    fn backend(&self) -> Option<Box<dyn Backend>> {
        Some(self.2.clone_boxed())
    }
}

pub fn eval<T: EvalArgs>(args: T) {
    let mut tape = BTreeSet::new();
    for output in args.outputs() {
        topological_sort(&mut tape, output);
    }
    for t in tape.into_iter() {
        {
            let backend = args.backend().unwrap_or(t.backend().clone_boxed());
            let backend = backend.as_ref();
            let primitive = t.primitive().clone_boxed();
            let primitive = primitive.as_ref();
            let inputs = &*t.inputs();
            let rule = eval_rule(backend, primitive).unwrap_or_else(|| {
                panic!(
                    "no eval rule for backend: {:?}, primitive: {:?}",
                    backend, primitive
                )
            });
            rule.eval(backend, primitive, inputs, &t);
        }
        if !args.retain_graph() {
            t.detach();
        }
    }
}
