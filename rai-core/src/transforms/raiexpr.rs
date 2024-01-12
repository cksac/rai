use crate::{backend::RaiExpr, eval, Differentiable, Func};

#[derive(Clone)]
pub struct RaiExprFunc<IN, OUT, F>
where
    F: Func<IN, OUT>,
    IN: Differentiable,
    OUT: Differentiable,
{
    func: F,
    phantom: std::marker::PhantomData<(IN, OUT)>,
}

impl<IN, OUT, F> RaiExprFunc<IN, OUT, F>
where
    F: Func<IN, OUT>,
    IN: Differentiable,
    OUT: Differentiable,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn raiexpr_of(&self, input: IN) -> String {
        let in_tensors = input.tensors();
        let output = self.func.apply(input);
        let out_tensors = output.tensors();
        eval(((in_tensors, out_tensors), true, RaiExpr));
        "todo raiexpr_of".to_string()
    }
}

pub fn raiexpr<IN, OUT, F>(func: F) -> RaiExprFunc<IN, OUT, F>
where
    F: Func<IN, OUT>,
    IN: Differentiable,
    OUT: Differentiable,
{
    RaiExprFunc::new(func)
}
