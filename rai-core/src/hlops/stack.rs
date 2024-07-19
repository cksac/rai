use crate::{Dim, Shape, Tensor};

#[track_caller]
pub fn stack<T: AsRef<Tensor>>(tensors: &[T], dim: impl Dim) -> Tensor {
    let t1 = tensors[0].as_ref().clone();
    let dim = t1.dim(dim) + 1;
    let inputs = tensors
        .iter()
        .map(|t| t.as_ref().unsqueeze(dim))
        .collect::<Vec<_>>();
    Tensor::cat(inputs.as_slice(), dim)
}

impl Tensor {
    #[inline]
    #[track_caller]
    pub fn stack<T: AsRef<Tensor>>(tensors: &[T], dim: impl Dim) -> Tensor {
        stack(tensors, dim)
    }
}

#[test]
fn test_stack() {
    use crate::{Cpu, Tensor, F32};
    let t1 = Tensor::rand([2, 3], F32, Cpu);
    let t2 = Tensor::rand([2, 3], F32, Cpu);
    let out = Tensor::stack(&[&t1, &t2], 0);
    println!("{}", out);

    // TODO: fix dim out of bound error
    // let out = Tensor::stack(&[&t1, &t2], 1);
    // println!("{}", out);
}
