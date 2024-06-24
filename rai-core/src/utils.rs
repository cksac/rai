use crate::{vjp, Error, Func, Shape, Tensor, TensorIter, F64};
use rustc_hash::FxHashSet;
use std::collections::{hash_map::RandomState, HashSet};

pub fn topological_sort<T>(outputs: &T) -> Result<Vec<Tensor>, Error>
where
    T: TensorIter,
{
    _topological_sort(outputs, |_| false, true)
}

pub fn topological_sort_filter<T, F>(outputs: &T, f: F) -> Result<Vec<Tensor>, Error>
where
    T: TensorIter,
    F: Fn(&Tensor) -> bool,
{
    _topological_sort(outputs, f, true)
}

pub fn topological_sort_unchecked<T>(outputs: &T) -> Vec<Tensor>
where
    T: TensorIter,
{
    _topological_sort(outputs, |_| false, false).unwrap()
}

pub fn topological_sort_filter_unchecked<T, F>(outputs: &T, f: F) -> Vec<Tensor>
where
    T: TensorIter,
    F: Fn(&Tensor) -> bool,
{
    _topological_sort(outputs, f, false).unwrap()
}

fn _topological_sort<T, F>(outputs: &T, f: F, check_err: bool) -> Result<Vec<Tensor>, Error>
where
    T: TensorIter,
    F: Fn(&Tensor) -> bool,
{
    let mut tape = Vec::new();
    let mut stack = Vec::new();
    let mut visited = FxHashSet::default();
    for o in outputs.tensor_iter() {
        stack.push((o.clone(), o.is_empty_inputs()));
        while let Some((t, visited_inputs)) = stack.pop() {
            if visited.contains(&t.id()) || f(&t) {
                continue;
            }
            if check_err {
                if let Some(err) = t.op().err().cloned() {
                    return Err(err.into());
                }
            }
            if visited_inputs {
                visited.insert(t.id());
                tape.push(t);
            } else {
                stack.push((t.clone(), true));
                for input in t.inputs().iter() {
                    stack.push((input.clone(), input.is_empty_inputs()));
                }
            }
        }
    }
    Ok(tape)
}

pub fn dprint<T: TensorIter>(args: T) {
    println!("{}", dot_graph(args));
}

pub fn dot_graph<T: TensorIter>(args: T) -> String {
    let output_set: Vec<usize> = args.tensor_iter().map(Tensor::id).collect();
    let tape = topological_sort_unchecked(&args);
    let nodes: HashSet<String, RandomState> = HashSet::from_iter(tape.iter().map(|tensor| {
        let color = if output_set.contains(&tensor.id()) {
            " color=\"red\""
        } else {
            ""
        };
        format!(
            "{} [label=\"{}: {}|{{dtype:|shape:|inputs:}}|{{{{{:?}}}|{{{:?}}}|{{{:?}}}}}\"{}];",
            tensor.id(),
            tensor.id(),
            tensor.op().dot_label(),
            tensor.dtype(),
            tensor.shape(),
            tensor.inputs().iter().map(|t| t.id()).collect::<Vec<_>>(),
            color
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

pub fn accelerate_enabled() -> bool {
    cfg!(feature = "accelerate")
}

pub fn mkl_enabled() -> bool {
    cfg!(feature = "mkl")
}

pub fn cuda_enabled() -> bool {
    cfg!(feature = "cuda")
}

pub fn metal_enabled() -> bool {
    cfg!(feature = "metal")
}

pub fn numerical_jvp<K, F>(
    func: F,
    input: impl AsRef<Tensor>,
    tangent: impl AsRef<Tensor>,
    eps: f64,
) -> Tensor
where
    F: for<'a> Func<K, &'a Tensor, Tensor>,
{
    let input = input.as_ref();
    let tangent = tangent.as_ref();
    let delta = &(tangent * eps);
    let f_pos = func.invoke(&(input + delta));
    let f_neg = func.invoke(&(input - delta));
    (f_pos - f_neg) * (0.5 / eps)
}

pub fn check_grad<K, F>(func: F, input: impl AsRef<Tensor>, eps: f64)
where
    F: for<'a> Func<K, &'a Tensor, Tensor> + Clone,
{
    check_vjp(func, input, eps);
    //todo: check jvp
}

pub fn check_vjp<K, F>(func: F, input: impl AsRef<Tensor>, eps: f64)
where
    F: for<'a> Func<K, &'a Tensor, Tensor> + Clone,
{
    let input = input.as_ref();
    let (v_out, vjp_fn) = vjp(func.clone(), input);
    let tangent = &input.rand_like();
    let tangent_out = &numerical_jvp(func, input, tangent, eps);
    let cotangent = &v_out.rand_like();
    let cotangent_out = &vjp_fn(cotangent.clone());

    let tangent = &tangent.flatten(..);
    let tangent_out = &tangent_out.flatten(..);
    let cotangent = &cotangent.flatten(..);
    let cotangent_out = &cotangent_out.flatten(..);

    let ip = tangent.matmul(cotangent_out);
    let ip_expected = tangent_out.matmul(cotangent);
    assert_all_close(&ip, &ip_expected, eps);
}

pub fn assert_all_close(x: &Tensor, y: &Tensor, eps: f64) {
    let diff = x - y;
    let t = diff.full_like(eps);
    let check = diff.gt(t);
    let r = check.to_dtype(F64).sum(..).as_scalar(F64).unwrap();
    assert_eq!(r, 0.0, "diff too large, x: {}, y: {}", x, y)
}
