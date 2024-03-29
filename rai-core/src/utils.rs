use crate::{ty_kind, vjp, Func, Shape, Tensor, TensorIter, F64};
use std::collections::{hash_map::RandomState, HashSet};

pub fn dprint<T: TensorIter>(args: T) {
    println!("{}", dot_graph(args));
}

pub fn dot_graph<T: TensorIter>(args: T) -> String {
    let mut tape = Vec::new();
    let mut output_set = HashSet::new();
    for output in args.tensor_iter() {
        output_set.insert(output.id());
        depth_first_traversal(&mut tape, output);
    }

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
            tensor.primitive().dot_label(),
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

fn depth_first_traversal(tape: &mut Vec<Tensor>, tensor: &Tensor) {
    if tape.contains(tensor) {
        return;
    }
    for input in tensor.inputs().iter() {
        depth_first_traversal(tape, input);
    }
    tape.push(tensor.clone());
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

pub fn numerical_jvp<F>(
    func: F,
    input: impl AsRef<Tensor>,
    tangent: impl AsRef<Tensor>,
    eps: f64,
) -> Tensor
where
    F: for<'a> Func<ty_kind::Basic, &'a Tensor, Tensor>,
{
    let input = input.as_ref();
    let tangent = tangent.as_ref();
    let delta = &(tangent * eps);
    let f_pos = func.apply(&(input + delta));
    let f_neg = func.apply(&(input - delta));
    (f_pos - f_neg) * (0.5 / eps)
}

pub fn check_grad<F>(func: F, input: impl AsRef<Tensor>, eps: f64)
where
    F: for<'a> Func<ty_kind::Basic, &'a Tensor, Tensor> + Clone,
{
    check_vjp(func, input, eps);
    //todo: check jvp
}

pub fn check_vjp<F>(func: F, input: impl AsRef<Tensor>, eps: f64)
where
    F: for<'a> Func<ty_kind::Basic, &'a Tensor, Tensor> + Clone,
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
    let r = check.to_dtype(F64).sum(..).as_scalar(F64);
    assert_eq!(r, 0.0, "diff too large, x: {}, y: {}", x, y)
}
