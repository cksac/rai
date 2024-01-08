use std::{
    collections::{hash_map::RandomState, HashSet, BTreeMap, HashMap},
    fmt::Debug,
    ops::Deref,
};

use crate::{Shape, Tensor};

pub trait TensorIter: Debug {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor>;
}

impl TensorIter for Tensor {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        std::iter::once(self)
    }
}

impl TensorIter for &Tensor {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        std::iter::once(*self)
    }
}

impl<const N: usize> TensorIter for [Tensor; N] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter()
    }
}

impl<const N: usize> TensorIter for [&Tensor; N] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter().map(Deref::deref)
    }
}

impl<const N: usize> TensorIter for &[Tensor; N] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter()
    }
}

impl<const N: usize> TensorIter for &[&Tensor; N] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter().map(Deref::deref)
    }
}

impl TensorIter for Vec<Tensor> {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter()
    }
}

impl TensorIter for &Vec<Tensor> {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter()
    }
}

impl TensorIter for &[Tensor] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter()
    }
}

impl TensorIter for &[&Tensor] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter().map(Deref::deref)
    }
}

impl<const N: usize, const M: usize> TensorIter for ([Tensor; N], [Tensor; M]) {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.0.iter().chain(self.1.iter())
    }
}

impl<const N: usize, const M: usize> TensorIter for (&[Tensor; N], &[Tensor; M]) {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.0.iter().chain(self.1.iter())
    }
}

impl<const N: usize, const M: usize, const R: usize> TensorIter
    for (&[Tensor; N], [&[Tensor; M]; R])
{
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.0.iter().chain(self.1.iter().flat_map(|v| v.iter()))
    }
}

impl<const N: usize, const M: usize> TensorIter for (&[Tensor; N], &[&[Tensor; M]]) {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.0.iter().chain(self.1.iter().flat_map(|v| v.iter()))
    }
}

impl<const N: usize, const R: usize> TensorIter for [&[Tensor; N]; R] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter().flat_map(|v| v.iter())
    }
}

impl<const N: usize> TensorIter for &[&[Tensor; N]] {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        self.iter().flat_map(|v| v.iter())
    }
}

impl TensorIter for (Tensor, Vec<Tensor>) {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        std::iter::once(&self.0).chain(self.1.iter())
    }
}

impl<const N: usize> TensorIter for (Tensor, [Tensor; N]) {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        std::iter::once(&self.0).chain(self.1.iter())
    }
}

impl TensorIter for (Tensor, BTreeMap<usize, Tensor>) {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        std::iter::once(&self.0).chain(self.1.values())
    }
}

impl TensorIter for (Tensor, HashMap<usize, Tensor>) {
    fn tensor_iter(&self) -> impl Iterator<Item = &Tensor> {
        std::iter::once(&self.0).chain(self.1.values())
    }
}

pub fn dot_graph<T: TensorIter>(args: T) -> String {
    let mut tape = Vec::new();
    for output in args.tensor_iter() {
        depth_first_traversal(&mut tape, output);
    }

    let nodes: HashSet<String, RandomState> = HashSet::from_iter(tape.iter().map(|tensor| {
        format!(
            "{} [label=\"{}: {}|{{dtype:|shape:|inputs:}}|{{{{{:?}}}|{{{:?}}}|{{{:?}}}}}\"];",
            tensor.id(),
            tensor.id(),
            tensor.primitive().dot_label(),
            tensor.dtype(),
            tensor.shape(),
            tensor.inputs().iter().map(|t| t.id()).collect::<Vec<_>>(),
        )
    }));

    let mut dot = String::new();
    dot.push_str("digraph {\n");
    dot.push_str("  node [shape=record];\n");

    for node in nodes {
        dot.push_str(&format!("  {}\n", node));
    }

    for tensor in tape.iter() {
        for input in tensor.inputs().iter() {
            dot.push_str(&format!("  {:?} -> {:?};\n", input.id(), tensor.id(),));
        }
    }

    dot.push('}');
    dot
}

fn depth_first_traversal(tape: &mut Vec<Tensor>, tensor: &Tensor) {
    for input in tensor.inputs().iter() {
        depth_first_traversal(tape, input);
    }
    if tape.contains(tensor) {
        return;
    }
    tape.push(tensor.clone());
}
