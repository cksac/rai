use crate::{dispatch::eval_rule, utils::topological_sort_filter, TensorIter};

pub trait EvalArgs {
    fn outputs(&self) -> &impl TensorIter;
    fn retain_graph(&self) -> bool {
        false
    }
}

impl<T> EvalArgs for T
where
    T: TensorIter,
{
    fn outputs(&self) -> &impl TensorIter {
        self
    }
}

impl<T> EvalArgs for (T, bool)
where
    T: TensorIter,
{
    fn outputs(&self) -> &impl TensorIter {
        &self.0
    }

    fn retain_graph(&self) -> bool {
        self.1
    }
}
pub fn eval<T: EvalArgs>(args: T) {
    let tape = topological_sort_filter(args.outputs(), |t| !t.is_evaluated());
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
