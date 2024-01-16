use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    ops::Deref,
    rc::Rc,
    sync::atomic,
};

use crate::{
    eval,
    ops::{self, ArangeArgs, FlattenArgs, ReduceArgs, VarArgs},
    utils::{self, dot_graph},
    Backend, DType, Dim, DynDType, ElemType, Primitive, Shape,
};

pub trait TensorLike: Debug + Display {
    fn as_any(&self) -> &dyn std::any::Any;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> &dyn DynDType;
}

struct TensorImpl {
    id: usize,
    backend: Box<dyn Backend>,
    dtype: Box<dyn DynDType>,
    shape: Vec<usize>,
    primitive: Box<dyn Primitive>,
    inputs: RefCell<Vec<Tensor>>,
    data: RefCell<Option<Box<dyn TensorLike>>>,
}

impl Tensor {
    pub fn new(
        backend: impl Into<Box<dyn Backend>>,
        dtype: impl Into<Box<dyn DynDType>>,
        shape: impl Shape,
        primitive: impl Into<Box<dyn Primitive>>,
        inputs: impl Into<Vec<Tensor>>,
    ) -> Self {
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        let id = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
        let inner = TensorImpl {
            id,
            backend: backend.into(),
            dtype: dtype.into(),
            shape: shape.shape().to_vec(),
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
    pub fn dtype(&self) -> &dyn DynDType {
        self.0.dtype.as_ref()
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
    pub fn full<T: ElemType>(
        val: T,
        shape: impl Shape,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::full::<T>(val, shape, backend)
    }

    #[inline]
    #[allow(unused_variables)]
    pub fn ones<D: DType>(
        shape: impl Shape,
        dtype: D,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::full::<D::Repr>(D::one(), shape, backend)
    }

    #[inline]
    #[allow(unused_variables)]
    pub fn zeros<D: DType>(
        shape: impl Shape,
        dtype: D,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::full::<D::Repr>(D::zero(), shape, backend)
    }

    #[inline]
    pub fn full_like<T: ElemType>(&self, val: T) -> Tensor {
        ops::full_like::<T>(self, val)
    }

    #[inline]
    pub fn zeros_like(&self) -> Tensor {
        ops::zeros_like(self)
    }

    #[inline]
    pub fn ones_like(&self) -> Tensor {
        ops::ones_like(self)
    }

    #[inline]
    pub fn normal(
        shape: impl Shape,
        dtype: impl DType,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::normal(shape, dtype, backend)
    }

    #[inline]
    pub fn arange<D: DType, T: ArangeArgs<D>>(
        args: T,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor
    where
        D::Repr:
            std::ops::Sub<D::Repr, Output = D::Repr> + std::ops::Div<D::Repr, Output = D::Repr>,
        D::Repr: Into<f64> + Copy,
    {
        ops::arange(args, backend)
    }

    #[inline]
    pub fn from_array<T: ElemType>(
        data: impl Into<Vec<T>> + Debug,
        shape: impl Shape,
        backend: impl Into<Box<dyn Backend>> + Debug,
    ) -> Tensor {
        ops::from_array(data, shape, backend)
    }

    #[inline]
    pub fn matmul<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::matmul(self, rhs.as_ref())
    }

    #[inline]
    pub fn eq<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::eq(self, rhs.as_ref())
    }

    #[inline]
    pub fn ne<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::ne(self, rhs.as_ref())
    }

    #[inline]
    pub fn gt<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::gt(self, rhs.as_ref())
    }

    #[inline]
    pub fn ge<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::ge(self, rhs.as_ref())
    }

    #[inline]
    pub fn lt<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::lt(self, rhs.as_ref())
    }

    #[inline]
    pub fn le<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::le(self, rhs.as_ref())
    }

    #[inline]
    pub fn maximum<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::maximum(self, rhs.as_ref())
    }

    #[inline]
    pub fn t(&self) -> Tensor {
        assert!(self.ndim() >= 2);
        let ndim = self.ndim();
        let mut dims = self.dims(..);
        dims.swap(ndim - 1, ndim - 2);
        ops::transpose(self, dims)
    }

    #[inline]
    pub fn transpose(&self, dims: impl Into<Vec<usize>> + Debug) -> Tensor {
        ops::transpose(self, dims)
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
    pub fn log(&self) -> Tensor {
        ops::log(self)
    }

    #[inline]
    pub fn log2(&self) -> Tensor {
        ops::log2(self)
    }

    #[inline]
    pub fn log10(&self) -> Tensor {
        ops::log10(self)
    }

    #[inline]
    pub fn sum<T: ReduceArgs>(&self, args: T) -> Tensor {
        ops::sum(self, args)
    }

    #[inline]
    pub fn max<T: ReduceArgs>(&self, args: T) -> Tensor {
        ops::max(self, args)
    }

    #[inline]
    pub fn min<T: ReduceArgs>(&self, args: T) -> Tensor {
        ops::min(self, args)
    }

    #[inline]
    pub fn mean<T: ReduceArgs>(&self, args: T) -> Tensor {
        ops::mean(self, args)
    }

    #[inline]
    pub fn var<T: VarArgs>(&self, args: T) -> Tensor {
        ops::var(self, args)
    }

    #[inline]
    pub fn gather(&self, dim: impl Dim, index: &Tensor) -> Tensor {
        ops::gather(self, dim, index)
    }

    #[inline]
    pub fn index_select(&self, dim: impl Dim, index: &Tensor) -> Tensor {
        ops::index_select(self, dim, index)
    }

    #[inline]
    pub fn as_type(&self, dtype: impl DType) -> Tensor {
        ops::as_type(self, dtype)
    }

    #[inline]
    pub fn as_type_of(&self, rhs: &Tensor) -> Tensor {
        ops::as_type_of(self, rhs)
    }

    #[inline]
    pub fn softmax<D: Dim>(&self, d: D) -> Tensor {
        ops::softmax(self, d)
    }

    #[inline]
    pub fn log_softmax<D: Dim>(&self, d: D) -> Tensor {
        ops::log_softmax(self, d)
    }

    #[inline]
    pub fn relu(&self) -> Tensor {
        ops::relu(self)
    }

    #[inline]
    pub fn flatten<T: FlattenArgs>(&self, args: T) -> Tensor {
        ops::flatten(self, args)
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
    #[track_caller]
    pub fn replace_data(&self, rhs: Tensor) {
        assert!(
            self.shape_eq(&rhs),
            "{:?} not align with rhs shape: {:?}\n{}",
            self,
            rhs.shape(),
            dot_graph([self, &rhs])
        );
        assert!(self.dtype() == rhs.dtype(), "dtype must be equal");
        assert!(self.backend() == rhs.backend(), "backend must be equal");
        assert!(rhs.is_evaluated(), "rhs must be evaluated");
        self.0.data.replace(rhs.0.data.take());
        self.detach();
    }

    #[inline]
    pub fn set_data<T: TensorLike + 'static>(&self, data: T) {
        assert!(
            self.shape_eq(&data.shape()),
            "{:?} not align with data shape: {:?}\n{}",
            self,
            data.shape(),
            self.dot_graph()
        );
        assert!(
            self.dtype() == data.dtype(),
            "{:?} not align with data dtype: {:?}\n{}",
            self,
            data.dtype(),
            self.dot_graph()
        );
        self.0.data.replace(Some(Box::new(data)));
    }

    #[inline]
    pub fn get_data<T>(&self) -> Option<impl Deref<Target = T> + '_>
    where
        T: 'static,
    {
        if self.is_evaluated() {
            Some(Ref::map(self.0.data.borrow(), |v| {
                v.as_ref().unwrap().as_any().downcast_ref::<T>().unwrap()
            }))
        } else {
            None
        }
    }

    #[inline]
    pub fn is_evaluated(&self) -> bool {
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
            .field("shape", &self.shape().shape())
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
        if !self.is_evaluated() {
            eval((self, true));
        }
        let data = self.0.data.borrow();
        f.debug_struct("Tensor")
            .field("id", &self.id())
            .field("shape", &self.shape().shape())
            .field("dtype", &self.dtype())
            .field("backend", &self.backend())
            .field("primitive", &self.primitive())
            .field(
                "inputs",
                &self.inputs().iter().map(|v| v.id()).collect::<Vec<_>>(),
            )
            .field("data", &format_args!("{}", data.as_deref().unwrap()))
            .finish()
    }
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

impl Shape for Tensor {
    fn shape(&self) -> &[usize] {
        self.0.shape.shape()
    }
}
