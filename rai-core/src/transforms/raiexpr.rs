use std::collections::{BTreeSet, HashMap};

use crate::{Differentiable, Func, Shape, Tensor, TensorIter};

pub fn raiexpr<IN, OUT, F>(func: &F, input: IN) -> String
where
    F: Func<IN, OUT>,
    IN: Differentiable,
    OUT: Differentiable,
{
    let mut id_seq = 0;
    let mut id_map: HashMap<usize, usize> = HashMap::new();

    let in_tensors = input.tensors();
    let output = func.apply(input);
    let out_tensors = output.tensors();

    let mut tape = BTreeSet::new();
    let input_set = in_tensors.tensor_iter().cloned().collect::<BTreeSet<_>>();
    fn recurse(tape: &mut BTreeSet<Tensor>, inputs: &BTreeSet<Tensor>, t: &Tensor) {
        for input in t.inputs().iter() {
            recurse(tape, inputs, input);
        }
        if tape.contains(t) || inputs.contains(t) {
            return;
        }
        tape.insert(t.clone());
    }

    for output in out_tensors.tensor_iter() {
        recurse(&mut tape, &input_set, output);
    }

    let inputs = in_tensors
        .tensor_iter()
        .map(|t| {
            format!(
                "%{}:{:?}{:?}",
                id_map.entry(t.id()).or_insert_with(|| {
                    id_seq += 1;
                    id_seq
                }),
                t.dtype(),
                t.shape()
            )
        })
        .collect::<Vec<String>>()
        .join(", ");

    let outputs = out_tensors
        .tensor_iter()
        .map(|t| format!("{:?}{:?}", t.dtype(), t.shape()))
        .collect::<Vec<String>>()
        .join(", ");

    let body = tape
        .iter()
        .map(|t| {
            let id = id_map
                .entry(t.id())
                .or_insert_with(|| {
                    id_seq += 1;
                    id_seq
                })
                .clone();
            format!(
                "\t%{}:{:?}{:?} = {} {}",
                id,
                t.dtype(),
                t.shape(),
                t.primitive().dot_label(),
                t.inputs()
                    .iter()
                    .map(|v| format!(
                        "%{}",
                        id_map.entry(v.id()).or_insert_with(|| {
                            id_seq += 1;
                            id_seq
                        }),
                    ))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    let returns = out_tensors
        .tensor_iter()
        .map(|t| {
            format!(
                "%{}",
                id_map.entry(t.id()).or_insert_with(|| {
                    id_seq += 1;
                    id_seq
                }),
            )
        })
        .collect::<Vec<String>>()
        .join(", ");

    format!(
        "fn({}) -> ({}) {{\n{}\n\treturn ({})\n}}",
        inputs, outputs, body, returns
    )
}
