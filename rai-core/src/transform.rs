use crate::{Backend, DType, Tensor};
use std::collections::HashMap;
use std::fmt::Debug;

pub trait Func<IN, OUT> {
    type Tangent: WithTensors + FromTensors;
    type Cotangent: WithTensors + FromTensors;
    fn call(&self, input: IN) -> OUT;
    fn captured_inputs(&self) -> Option<Vec<Tensor>> {
        None
    }
}

pub trait WithTensors {
    fn tensors(&self) -> Vec<Tensor>;
}

pub trait FromTensors {
    fn from_tensors(tensors: Vec<Tensor>) -> Self;
}

pub trait Input: WithTensors {}

pub trait Output: WithTensors {}

pub fn jvp<IN, OUT, F>(func: F, input: IN, tangents: &F::Tangent) -> (OUT, F::Cotangent)
where
    F: Func<IN, OUT>,
    IN: Input,
    OUT: Output,
{
    let mut tangent_map = HashMap::new();
    let params = if let Some(params) = func.captured_inputs() {
        params
    } else {
        input.tensors()
    };

    for (p, t) in params.iter().zip(tangents.tensors()) {
        tangent_map.insert(p.id(), t.clone());
    }
    let output = func.call(input);

    let mut jvps = Vec::new();
    for t in output.tensors() {
        jvps.push(t.jvp(&mut tangent_map));
    }
    let jvps = F::Cotangent::from_tensors(jvps);

    (output, jvps)
}

type VjpFn<IN, OUT> = Box<dyn Fn(IN) -> OUT>;

pub fn vjp<IN, OUT, F>(func: F, input: IN) -> (OUT, VjpFn<F::Cotangent, F::Tangent>)
where
    F: Func<IN, OUT>,
    IN: Input,
    OUT: Output,
{
    let input_ids: Vec<usize> = input.tensors().iter().map(|v| v.id()).collect();
    let output = func.call(input);
    let out_tensors = output.tensors().iter().cloned().collect::<Vec<Tensor>>();

    let vjps_fn = move |cotangents: F::Cotangent| {
        let mut cotangent_map = HashMap::new();

        for (p, c) in out_tensors.iter().zip(cotangents.tensors()) {
            cotangent_map.insert(p.id(), c);
        }

        for t in out_tensors.iter() {
            t.vjp(&mut cotangent_map);
        }

        let mut vjps: Vec<Tensor> = Vec::with_capacity(input_ids.len());
        for (i, id) in input_ids.iter().enumerate() {
            vjps[i] = cotangent_map.get(id).unwrap().clone();
        }
        F::Tangent::from_tensors(vjps)
    };

    (output, Box::new(vjps_fn))
}

// impls
impl<F> Func<&[Tensor; 1], [Tensor; 1]> for F
where
    F: Fn(&Tensor) -> Tensor,
{
    type Tangent = [Tensor; 1];
    type Cotangent = [Tensor; 1];
    fn call(&self, input: &[Tensor; 1]) -> [Tensor; 1] {
        [self(&input[0])]
    }
}

impl<F> Func<&[Tensor; 2], [Tensor; 1]> for F
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    type Tangent = [Tensor; 2];
    type Cotangent = [Tensor; 1];
    fn call(&self, input: &[Tensor; 2]) -> [Tensor; 1] {
        [self(&input[0], &input[1])]
    }
}

impl<const N: usize> WithTensors for [Tensor; N] {
    fn tensors(&self) -> Vec<Tensor> {
        self.to_vec()
    }
}

impl<const N: usize> WithTensors for &[Tensor; N] {
    fn tensors(&self) -> Vec<Tensor> {
        self.to_vec()
    }
}

impl WithTensors for Vec<Tensor> {
    fn tensors(&self) -> Vec<Tensor> {
        self.clone()
    }
}

impl WithTensors for Tensor {
    fn tensors(&self) -> Vec<Tensor> {
        vec![self.clone()]
    }
}

impl WithTensors for &Tensor {
    fn tensors(&self) -> Vec<Tensor> {
        vec![(*self).clone()]
    }
}

impl<const N: usize> FromTensors for [Tensor; N] {
    fn from_tensors(tensors: Vec<Tensor>) -> Self {
        tensors.try_into().unwrap_or_else(|v: Vec<Tensor>| {
            panic!("Expected a Vec of length {} but it was {}", N, v.len())
        })
    }
}

impl FromTensors for Vec<Tensor> {
    fn from_tensors(tensors: Vec<Tensor>) -> Self {
        tensors
    }
}

impl<const N: usize> Input for [Tensor; N] {}

impl<const N: usize> Output for [Tensor; N] {}

impl<const N: usize> Input for &[Tensor; N] {}

impl<const N: usize> Output for &[Tensor; N] {}

impl Input for Vec<Tensor> {}
impl Input for Tensor {}
impl Input for &Tensor {}
impl Output for Tensor {}

// module impl
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        has_bias: bool,
        dtype: DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Self {
        let backend = backend.into();
        // TODO: init strategy
        let weight = Tensor::normal([out_features, in_features], dtype, backend.clone());
        let bias = if has_bias {
            Some(Tensor::normal([out_features], dtype, backend))
        } else {
            None
        };
        Self { weight, bias }
    }
}

impl Func<&Tensor, Tensor> for Linear {
    type Tangent = Vec<Tensor>;
    type Cotangent = Vec<Tensor>;

    fn call(&self, input: &Tensor) -> Tensor {
        match &self.bias {
            Some(bias) => self.weight.matmul(input) + bias,
            None => input.matmul(&self.weight),
        }
    }

    fn captured_inputs(&self) -> Option<Vec<Tensor>> {
        Some(match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{backend::Cpu, transform::jvp, DType, Tensor};

    use super::Linear;

    fn func(a: &Tensor, b: &Tensor) -> Tensor {
        a + b
    }

    #[test]
    fn test_jvp() {
        let backend = &Cpu;
        let a = Tensor::full(1.0, [2, 3], DType::F32, backend);
        let b = Tensor::full(1.0, [2, 3], DType::F32, backend);

        let at = Tensor::full(1.0, [2, 3], DType::F32, backend);
        let bt = Tensor::full(3.0, [2, 3], DType::F32, backend);

        let (output, jvps) = jvp(func, &[a, b], &[at, bt]);
    }

    #[test]
    fn test_module() {
        let backend = &Cpu;

        let linear = Linear::new(100, 10, true, DType::F32, backend);
        let input = Tensor::normal([100], DType::F32, backend);

        let wt = linear.weight.ones_like();
        let bt = linear.bias.as_ref().unwrap().ones_like();

        let (output, jvps) = jvp(linear, &input, &vec![wt, bt]);
    }
}
