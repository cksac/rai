use crate::{utils::topological_sort_filter, Func, Result, Shape, Tensor, TensorIter, Value};
use colored::*;
use std::collections::HashMap;

pub fn raiexpr<K, IN, OUT, F>(func: &F, input: IN) -> Result<String>
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
            t.op().dot_label(),
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
            t.op().dot_label(),
            t.inputs()
                .iter()
                .map(|v| decl(id_map, id_seq, v))
                .collect::<Vec<_>>()
                .join(" "),
            t.location()
        )
    }
    let input_set: Vec<usize> = in_tensors.tensor_iter().map(Tensor::id).collect();
    let tape = topological_sort_filter(&out_tensors, |t| input_set.contains(&t.id()))?;

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

    Ok(format!(
        "fn({}) -> ({}) {{\n{}\n\treturn ({})\n}}",
        inputs, outputs, body, returns
    ))
}
