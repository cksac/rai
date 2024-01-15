use std::{
    collections::BTreeSet,
    fmt::{format, Display},
};

use crate::{utils::topological_sort, Differentiable, Func, Shape, Tensor, TensorIter};

#[derive(Debug)]
pub struct RaiFunc {
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    expressions: Vec<Tensor>,
}

impl RaiFunc {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            expressions: Vec::new(),
        }
    }
}

impl Display for RaiFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inputs = self
            .inputs
            .iter()
            .map(|t| format!("%{}:{:?}{:?}", t.id(), t.dtype(), t.shape()))
            .collect::<Vec<String>>()
            .join(", ");
        let outputs = self
            .outputs
            .iter()
            .map(|t| format!("%{}:{:?}{:?}", t.id(), t.dtype(), t.shape()))
            .collect::<Vec<String>>()
            .join(", ");
        let body = self
            .expressions
            .iter()
            .map(|t| {
                format!(
                    "\t%{}:{:?}{:?} = {} {}",
                    t.id(),
                    t.dtype(),
                    t.shape(),
                    t.primitive().dot_label(),
                    t.inputs()
                        .iter()
                        .map(|v| format!("%{}", v.id()))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            })
            .collect::<Vec<String>>()
            .join("\n");
        f.write_fmt(format_args!(
            "fn({}) -> ({}) {{\n{}\n}}",
            inputs, outputs, body
        ))
    }
}

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

    pub fn raiexpr_of(&self, input: IN) -> RaiFunc {
        let mut ir_func = RaiFunc::new();
        let in_tensors = input.tensors();
        for t in in_tensors.tensor_iter() {
            ir_func.inputs.push(t.clone());
        }

        let output = self.func.apply(input);
        let out_tensors = output.tensors();
        for t in out_tensors.tensor_iter() {
            ir_func.outputs.push(t.clone());
        }

        let mut tape = BTreeSet::new();
        for output in out_tensors.tensor_iter() {
            topological_sort(&mut tape, output);
        }
        for t in tape.into_iter() {
            ir_func.expressions.push(t.clone())
        }

        ir_func
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
