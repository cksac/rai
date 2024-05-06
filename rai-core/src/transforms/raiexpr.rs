use crate::{Func, Shape, Tensor, TensorIter, Value};
use colored::*;
use std::collections::{BTreeSet, HashMap};

pub fn raiexpr<K, IN, OUT, F>(func: &F, input: IN) -> String
where
    F: Func<K, IN, OUT>,
    IN: Value,
    OUT: Value,
{
    let mut id_seq = 0;
    let mut id_map: HashMap<usize, usize> = HashMap::new();

    let in_tensors = input.tensors();
    let output = func.invoke(input);
    let out_tensors = output.tensors();

    fn id(id_map: &mut HashMap<usize, usize>, id_seq: &mut usize, t: &Tensor) -> String {
        let id = *id_map.entry(t.id()).or_insert_with(|| {
            *id_seq += 1;
            *id_seq
        });
        format!("%{}", id).yellow().to_string()
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

    #[cfg(not(feature = "debug-location"))]
    fn expr(id_map: &mut HashMap<usize, usize>, id_seq: &mut usize, t: &Tensor) -> String {
        format!(
            "\t{} = {} {}",
            decl(id_map, id_seq, t),
            t.primitive().dot_label(),
            t.inputs()
                .iter()
                .map(|v| decl(id_map, id_seq, v))
                .collect::<Vec<_>>()
                .join(" ")
        )
    }

    #[cfg(feature = "debug-location")]
    fn expr(id_map: &mut HashMap<usize, usize>, id_seq: &mut usize, t: &Tensor) -> String {
        format!(
            "\t{} = {} {} // {}",
            decl(id_map, id_seq, t),
            t.primitive().dot_label(),
            t.inputs()
                .iter()
                .map(|v| decl(id_map, id_seq, v))
                .collect::<Vec<_>>()
                .join(" "),
            t.location()
        )
    }

    // use iterative instead of recursive to avoid stack overflow
    // TODO: use proper topo sort algorithm, now sort by id in BTreeSet
    let input_set = in_tensors.tensor_iter().cloned().collect::<BTreeSet<_>>();
    let mut tape = BTreeSet::new();
    let mut stack = Vec::new();
    for output in out_tensors.tensor_iter() {
        stack.push(output.clone());
    }

    while let Some(t) = stack.pop() {
        if tape.contains(&t) || input_set.contains(&t) {
            continue;
        }
        tape.insert(t.clone());
        for input in t.inputs().iter() {
            stack.push(input.clone());
        }
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
        .map(|t| expr(&mut id_map, &mut id_seq, t))
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
