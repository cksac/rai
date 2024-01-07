use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic;

use crate::ops::{ArangeArgs, ReduceSumArgs};
use crate::{eval, utils, AsDim};
use crate::{ops, Backend, DType, Primitive, Shape};

pub trait TensorLike: Debug + Display {
    fn as_any(&self) -> &dyn std::any::Any;
    fn shape(&self) -> &[usize];
}

struct TensorImpl {
    id: usize,
    backend: Box<dyn Backend>,
    dtype: DType,
    shape: Vec<usize>,
    primitive: Box<dyn Primitive>,
    inputs: RefCell<Vec<Tensor>>,
    data: RefCell<Option<Box<dyn TensorLike>>>,
}

impl Tensor {
    pub fn new(
        backend: impl Into<Box<dyn Backend>>,
        dtype: DType,
        shape: impl Shape,
        primitive: impl Into<Box<dyn Primitive>>,
        inputs: impl Into<Vec<Tensor>>,
    ) -> Self {
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        let id = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);

        let inner = TensorImpl {
            id,
            backend: backend.into(),
            dtype,
            shape: shape.to_vec(),
            primitive: primitive.into(),
            inputs: RefCell::new(inputs.into()),
            data: RefCell::new(None),
        };
        Tensor(Rc::new(inner))
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.0.id
    }

    #[inline]
    pub fn backend(&self) -> &dyn Backend {
        self.0.backend.as_ref()
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    #[inline]
    pub fn shape(&self) -> impl Shape + '_ {
        &self.0.shape
    }

    #[inline]
    pub fn primitive(&self) -> &dyn Primitive {
        self.0.primitive.as_ref()
    }

    #[inline]
    pub fn inputs(&self) -> impl Deref<Target = [Tensor]> + '_ {
        Ref::map(self.0.inputs.borrow(), |v| v.as_slice())
    }

    #[inline]
    pub fn full(
        val: f64,
        shape: impl Shape,
        dtype: DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::full(val, shape, dtype, backend)
    }

    #[inline]
    pub fn ones(
        shape: impl Shape,
        dtype: DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::full(1.0, shape, dtype, backend)
    }

    #[inline]
    pub fn zeros(
        shape: impl Shape,
        dtype: DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::full(0.0, shape, dtype, backend)
    }

    #[inline]
    pub fn full_like(&self, val: f64) -> Tensor {
        ops::full(val, self.shape(), self.dtype(), self.backend())
    }

    #[inline]
    pub fn zeros_like(&self) -> Tensor {
        ops::full(0.0, self.shape(), self.dtype(), self.backend())
    }

    #[inline]
    pub fn ones_like(&self) -> Tensor {
        ops::full(1.0, self.shape(), self.dtype(), self.backend())
    }

    #[inline]
    pub fn normal(
        shape: impl Shape,
        dtype: DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::normal(shape, dtype, backend)
    }

    #[inline]
    pub fn arange<T: ArangeArgs>(args: T, backend: impl Into<Box<dyn Backend>> + Debug) -> Tensor {
        ops::arange(args, backend)
    }

    #[inline]
    pub fn matmul<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::matmul(self, rhs.as_ref())
    }

    #[inline]
    pub fn greater<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::greater(self, rhs.as_ref())
    }

    #[inline]
    pub fn greater_equal<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::greater_equal(self, rhs.as_ref())
    }

    #[inline]
    pub fn less<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::less(self, rhs.as_ref())
    }

    #[inline]
    pub fn less_equal<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::less_equal(self, rhs.as_ref())
    }

    #[inline]
    pub fn maximum<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::maximum(self, rhs.as_ref())
    }

    #[inline]
    pub fn t(&self) -> Tensor {
        assert!(self.ndim() >= 2);
        let ndim = self.ndim();
        let mut axes = Vec::from_iter(0..ndim);
        axes.swap(ndim - 1, ndim - 2);
        ops::transpose(self, axes)
    }

    #[inline]
    pub fn transpose(&self, axes: impl Into<Vec<usize>> + Debug) -> Tensor {
        ops::transpose(self, axes)
    }

    #[inline]
    pub fn broadcast_to(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_to(self, shape)
    }

    #[inline]
    pub fn reshape(&self, shape: impl Shape) -> Tensor {
        ops::reshape(self, shape)
    }

    #[inline]
    pub fn add<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Add<T, Output = Tensor>,
    {
        std::ops::Add::add(self, rhs)
    }

    #[inline]
    pub fn mul<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Mul<T, Output = Tensor>,
    {
        std::ops::Mul::mul(self, rhs)
    }

    #[inline]
    pub fn sin(&self) -> Tensor {
        ops::sin(self)
    }

    #[inline]
    pub fn cos(&self) -> Tensor {
        ops::cos(self)
    }

    #[inline]
    pub fn square(&self) -> Tensor {
        ops::square(self)
    }

    #[inline]
    pub fn sqrt(&self) -> Tensor {
        ops::sqrt(self)
    }

    #[inline]
    pub fn rsqrt(&self) -> Tensor {
        ops::rsqrt(self)
    }

    #[inline]
    pub fn sign(&self) -> Tensor {
        ops::sign(self)
    }

    #[inline]
    pub fn abs(&self) -> Tensor {
        ops::abs(self)
    }

    #[inline]
    pub fn exp(&self) -> Tensor {
        ops::exp(self)
    }

    #[inline]
    pub fn softmax<T: AsDim>(&self, dim: T) -> Tensor {
        ops::softmax(self, dim.as_dim(self))
    }

    #[inline]
    pub fn relu(&self) -> Tensor {
        ops::relu(self)
    }

    #[inline]
    pub fn sum(&self) -> Tensor {
        ops::reduce_sum(self, self.axes())
    }

    #[inline]
    pub fn reduce_sum<T: ReduceSumArgs>(&self, args: T) -> Tensor {
        ops::reduce_sum(self, args)
    }

    pub fn jvp(&self, tangent_cache: &mut HashMap<usize, Tensor>) -> Tensor {
        if let Some(tangent) = tangent_cache.get(&self.id()) {
            return tangent.clone();
        }
        let primals = &*self.inputs();
        let tangents = primals
            .iter()
            .map(|t| t.jvp(tangent_cache))
            .collect::<Vec<_>>();
        let tangent = self.primitive().jvp(self, primals, &tangents);
        tangent_cache.insert(self.id(), tangent.clone());
        tangent
    }

    pub fn vjp(&self, cotangent_cache: &mut HashMap<usize, Tensor>) {
        let cotan = &cotangent_cache
            .entry(self.id())
            .or_insert_with(|| self.ones_like())
            .clone();
        let primals = &*self.inputs();
        if primals.is_empty() {
            return;
        }

        let cotangents = self.primitive().vjp(self, primals, cotan);
        for (i, cotangent) in cotangents.into_iter().enumerate() {
            let id = primals[i].id();
            if let Some(sum) = cotangent_cache.get(&id) {
                cotangent_cache.insert(id, sum + cotangent);
            } else {
                cotangent_cache.insert(id, cotangent);
            }
        }
        for i in primals {
            i.vjp(cotangent_cache)
        }
    }

    #[inline]
    pub fn dot_graph(&self) -> String {
        utils::dot_graph(self)
    }

    #[inline]
    pub fn detach(&self) {
        self.0.inputs.borrow_mut().clear();
    }

    #[inline]
    pub fn replace_data(&self, rhs: Tensor) {
        assert!(
            self.shape_eq(&rhs),
            "{:?} not align with rhs shape: {:?},",
            self,
            rhs.shape()
        );
        assert!(self.dtype() == rhs.dtype(), "dtype must be equal");
        assert!(self.backend() == rhs.backend(), "backend must be equal");
        assert!(rhs.is_evalualted(), "rhs must be evaluated");
        self.0.data.replace(rhs.0.data.take());
        self.detach();
    }

    #[inline]
    pub fn set_data<T: TensorLike + 'static>(&self, data: T) {
        assert!(
            self.shape_eq(&data.shape()),
            "{:?} not align with data shape: {:?},",
            self,
            data.shape()
        );
        self.0.data.replace(Some(Box::new(data)));
    }

    #[inline]
    pub fn get_data<T>(&self) -> Option<impl Deref<Target = T> + '_>
    where
        T: 'static,
    {
        if self.is_evalualted() {
            Some(Ref::map(self.0.data.borrow(), |v| {
                v.as_ref().unwrap().as_any().downcast_ref::<T>().unwrap()
            }))
        } else {
            None
        }
    }

    #[inline]
    pub fn is_evalualted(&self) -> bool {
        self.0
            .data
            .borrow()
            .as_ref()
            .filter(|v| v.as_any().type_id() == self.backend().data_type_id())
            .is_some()
    }
}

#[derive(Clone)]
pub struct Tensor(Rc<TensorImpl>);

impl Hash for Tensor {
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        hasher.write_usize(self.id());
    }
}

impl PartialOrd for Tensor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.id.cmp(&other.0.id))
    }
}

impl Ord for Tensor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.id.cmp(&other.0.id)
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for Tensor {}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id())
            .field("shape", &self.shape().dims())
            .field("dtype", &self.dtype())
            .field("backend", &self.backend())
            .field("primitive", &self.primitive())
            .field(
                "inputs",
                &self.inputs().iter().map(|v| v.id()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.is_evalualted() {
            eval((self, true));
        }
        let data = self.0.data.borrow();
        f.write_fmt(format_args!("{}", data.as_deref().unwrap()))
    }
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Shape for Tensor {
    fn dims(&self) -> &[usize] {
        self.0.shape.dims()
    }

    fn ndim(&self) -> usize {
        self.0.shape.ndim()
    }
}

impl Shape for &Tensor {
    fn dims(&self) -> &[usize] {
        self.0.shape.dims()
    }

    fn ndim(&self) -> usize {
        self.0.shape.ndim()
    }
}
