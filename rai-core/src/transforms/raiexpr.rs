use crate::{Func, Shape, Tensor, TensorIter, Value};
use colored::*;
use std::collections::{BTreeSet, HashMap};

pub fn raiexpr<IN, OUT, F>(func: &F, input: IN) -> String
where
    F: Func<IN, OUT>,
    IN: Value,
    OUT: Value,
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

    fn id(id_map: &mut HashMap<usize, usize>, id_seq: &mut usize, t: &Tensor) -> String {
        let id = *id_map.entry(t.id()).or_insert_with(|| {
            *id_seq += 1;
            *id_seq
        });
        format!("%{}", id.to_string()).yellow().to_string()
    }

    fn decl(id_map: &mut HashMap<usize, usize>, id_seq: &mut usize, t: &Tensor) -> String {
        format!("{}:{}", id(id_map, id_seq, t), ty(t))
    }

    fn ty(t: &Tensor) -> String {
        format!(
            "{}{}",
            format!("{:?}", t.dtype()).to_lowercase().cyan(),
            format!("{:?}", t.shape()).purple(),
        )
    }

    for output in out_tensors.tensor_iter() {
        recurse(&mut tape, &input_set, output);
    }

    let inputs = in_tensors
        .tensor_iter()
        .map(|t| decl(&mut id_map, &mut id_seq, t))
        .collect::<Vec<String>>()
        .join(", ");

    let outputs = out_tensors
        .tensor_iter()
        .map(ty)
        .collect::<Vec<String>>()
        .join(", ");

    let body = tape
        .iter()
        .map(|t| {
            format!(
                "\t{} = {} {}",
                decl(&mut id_map, &mut id_seq, t),
                t.primitive().dot_label(),
                t.inputs()
                    .iter()
                    .map(|v| decl(&mut id_map, &mut id_seq, v))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    let returns = out_tensors
        .tensor_iter()
        .map(|t| id(&mut id_map, &mut id_seq, t))
        .collect::<Vec<String>>()
        .join(", ");

    format!(
        "fn({}) -> ({}) {{\n{}\n\treturn ({})\n}}",
        inputs, outputs, body, returns
    )
}
