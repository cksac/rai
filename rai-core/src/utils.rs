use std::collections::{hash_map::RandomState, HashSet};

use crate::{Shape, Tensor, TensorIter};

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
    if tape.contains(tensor) {
        return;
    }
    for input in tensor.inputs().iter() {
        depth_first_traversal(tape, input);
    }
    tape.push(tensor.clone());
}
