use crate::{Device, Primitive, Shape, Tensor, Type};
use std::any::Any;
use tracing::Level;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Broadcast {
    pub shape: Vec<usize>,
}
impl Broadcast {
    pub fn new(shape: impl Shape) -> Self {
        Self {
            shape: shape.shape().to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Primitive for Broadcast {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn dot_label(&self) -> String {
        format!("Broadcast({:?})", self.shape)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.broadcast_to(self.shape())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let shape = x.shape().to_vec();
        let diff = cotangent.ndim() - shape.ndim();
        let mut dims = Vec::new();
        for i in 0..cotangent.ndim() {
            if i < diff || shape[i - diff] != cotangent.shape_at(i) {
                dims.push(i);
            }
        }
        let cotangent_x = cotangent.sum((dims, true)).reshape(&shape);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reshape {
    pub shape: Vec<usize>,
}
impl Reshape {
    pub fn new(shape: impl Shape) -> Self {
        Self {
            shape: shape.shape().to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl Primitive for Reshape {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Reshape({:?})", self.shape)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.reshape(self.shape())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.reshape(x);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transpose {
    pub dim0: usize,
    pub dim1: usize,
}

impl Transpose {
    pub fn new(dim0: usize, dim1: usize) -> Self {
        Self { dim0, dim1 }
    }

    pub fn dim0(&self) -> usize {
        self.dim0
    }

    pub fn dim1(&self) -> usize {
        self.dim1
    }
}

impl Primitive for Transpose {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Transpose({}, {})", self.dim0(), self.dim1())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.transpose(self.dim0, self.dim1)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = cotangent.transpose(self.dim0, self.dim1);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToContiguous;

impl Primitive for ToContiguous {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.clone()
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, _primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let cotangent_x = cotangent.clone();
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug)]
pub struct ToDevice<D: Device + Clone> {
    pub device: D,
}

impl<D: Device + Clone> ToDevice<D> {
    pub fn new(device: D) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &D {
        &self.device
    }
}

impl<D: Device + Clone + 'static> Primitive for ToDevice<D> {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("ToDevice({:?})", self.device())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.to_device(self.device())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.to_device(x);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ToDType<D: Type> {
    pub dtype: D,
}

impl<D: Type> ToDType<D> {
    pub fn new(dtype: D) -> Self {
        Self { dtype }
    }
}

impl<D: Type> Primitive for ToDType<D> {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("ToDType({:?})", &self.dtype)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, _output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.to_dtype(self.dtype)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let x = &primals[0];
        let cotangent_x = cotangent.to_dtype(x);
        vec![cotangent_x]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permute {
    pub dims: Vec<usize>,
}
impl Permute {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

impl Primitive for Permute {
    fn clone_boxed(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dot_label(&self) -> String {
        format!("Permute({:?})", self.dims)
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn jvp(&self, output: &Tensor, _primals: &[Tensor], tangents: &[Tensor]) -> Tensor {
        let tangent_x = &tangents[0];
        tangent_x.permute(self.dims())
    }

    #[tracing::instrument(ret(level = Level::TRACE))]
    fn vjp(&self, _output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor> {
        let dims = self.dims();
        let mut inv_dims = vec![0; dims.len()];
        for (i, &dim_idx) in dims.iter().enumerate() {
            inv_dims[dim_idx] = i
        }
        let cotangent_x = cotangent.permute(inv_dims);
        vec![cotangent_x]
    }
}
