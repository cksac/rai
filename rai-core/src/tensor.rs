use crate::{
    eval,
    nn::{ApplyModule, Module},
    utils::{self, dot_graph},
    AsDType, AsDevice, DType, Device, GradMap, Op, Shape, Type,
};
#[cfg(feature = "debug-location")]
use std::panic::Location;
use std::{
    any::Any,
    borrow::Cow,
    cell::{Ref, RefCell},
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    ops::Deref,
    rc::Rc,
};

pub trait TensorLike: Debug + Display {
    fn as_any(&self) -> &dyn Any;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> &dyn DType;
    fn as_scalar(&self) -> Box<dyn Any>;
    fn as_vec(&self) -> Box<dyn Any>;
    fn as_bytes(&self) -> Vec<u8>;
}

struct TensorImpl {
    device: Box<dyn Device>,
    dtype: Box<dyn DType>,
    shape: Vec<usize>,
    op: Box<dyn Op>,
    inputs: RefCell<Vec<Tensor>>,
    data: RefCell<Option<Box<dyn TensorLike>>>,
    #[cfg(feature = "debug-location")]
    location: &'static Location<'static>,
}

impl Tensor {
    #[track_caller]
    pub fn new(
        device: impl AsDevice,
        dtype: impl AsDType,
        shape: impl Shape,
        op: impl Into<Box<dyn Op>>,
        inputs: impl Into<Vec<Tensor>>,
    ) -> Self {
        let inner = TensorImpl {
            device: device.into_boxed_device(),
            dtype: dtype.into_boxed_dtype(),
            shape: shape.shape().to_vec(),
            op: op.into(),
            inputs: RefCell::new(inputs.into()),
            data: RefCell::new(None),
            #[cfg(feature = "debug-location")]
            location: Location::caller(),
        };
        Tensor(Rc::new(inner))
    }

    #[inline]
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    #[inline]
    pub fn device(&self) -> &dyn Device {
        self.0.device.as_ref()
    }

    #[inline]
    pub fn dtype(&self) -> &dyn DType {
        self.0.dtype.as_ref()
    }

    #[inline]
    pub fn op(&self) -> &dyn Op {
        self.0.op.as_ref()
    }

    #[inline]
    pub fn is_empty_inputs(&self) -> bool {
        self.0.inputs.borrow().is_empty()
    }

    #[inline]
    pub fn inputs(&self) -> impl Deref<Target = [Tensor]> + '_ {
        Ref::map(self.0.inputs.borrow(), |v| v.as_slice())
    }

    #[inline]
    pub fn set_inputs(&self, inputs: Vec<Tensor>) {
        self.0.inputs.replace(inputs);
    }

    #[cfg(feature = "debug-location")]
    #[inline]
    pub fn location(&self) -> &'static Location<'static> {
        self.0.location
    }

    pub(crate) fn jvp(&self, tangent_cache: &mut GradMap) -> Tensor {
        if let Some(tangent) = tangent_cache.get(self.id()).cloned() {
            return tangent;
        }
        let primals = &*self.inputs();
        let tangents = primals
            .iter()
            .map(|t| t.jvp(tangent_cache))
            .collect::<Vec<_>>();
        let tangent = self.op().jvp(self, primals, &tangents);
        tangent_cache.insert(self.id(), tangent.clone());
        tangent
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
        debug_assert!(
            self.shape() == rhs.shape(),
            "{:?} not align with rhs shape: {:?}\n{}",
            self,
            rhs.shape(),
            dot_graph([self, &rhs])
        );
        debug_assert!(self.dtype() == rhs.dtype(), "dtype must be equal");
        debug_assert!(
            self.device() == rhs.device(),
            "lhs device {:?} not equal rhs device {:?}",
            self.device(),
            rhs.device()
        );
        if !rhs.is_evaluated() {
            eval(&rhs);
        }
        self.0.data.replace(rhs.0.data.take());
        self.detach();
    }

    #[inline]
    pub fn set_data<T: TensorLike + 'static>(&self, data: T) {
        assert!(
            self.shape() == data.shape(),
            "{:?} not align with data shape: {:?}",
            self,
            data.shape()
        );
        assert!(
            self.dtype() == data.dtype(),
            "{:?} not align with data dtype: {:?}",
            self,
            data.dtype()
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

    #[allow(unused_variables)]
    pub fn as_scalar<T: Type>(&self, dtype: T) -> T::Repr {
        if !self.is_evaluated() {
            eval(self);
        }
        let data = self.0.data.borrow();
        let data = data.as_deref().unwrap();
        *data.as_scalar().downcast_ref::<T::Repr>().unwrap()
    }

    #[allow(unused_variables)]
    pub fn as_vec<T: Type>(&self, dtype: T) -> Vec<T::Repr> {
        if !self.is_evaluated() {
            eval(self);
        }
        let data = self.0.data.borrow();
        let data = data.as_deref().unwrap();
        data.as_vec()
            .downcast_ref::<Vec<T::Repr>>()
            .unwrap()
            .clone()
    }

    #[inline]
    pub fn apply<M>(&self, module: M) -> M::Output
    where
        M: Module<Input = Self>,
        Self: ApplyModule<M>,
    {
        ApplyModule::apply(self, module)
    }

    #[inline]
    pub fn is_evaluated(&self) -> bool {
        self.0.data.borrow().as_ref().is_some()
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
        Some(self.id().cmp(&other.id()))
    }
}

impl Ord for Tensor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id().cmp(&other.id())
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for Tensor {}

#[cfg(not(feature = "debug-location"))]
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id())
            .field("shape", &self.shape().shape())
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("op", &self.op())
            .field(
                "inputs",
                &self.inputs().iter().map(|v| v.id()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(feature = "debug-location")]
impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id())
            .field("shape", &self.shape().shape())
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("op", &self.op())
            .field(
                "inputs",
                &self.inputs().iter().map(|v| v.id()).collect::<Vec<_>>(),
            )
            .field("location", &format_args!("{}", &self.0.location))
            .finish()
    }
}

#[cfg(not(feature = "debug-location"))]
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
            .field("device", &self.device())
            .field("op", &self.op())
            .field(
                "inputs",
                &self.inputs().iter().map(|v| v.id()).collect::<Vec<_>>(),
            )
            .field("data", &format_args!("{}", data.as_deref().unwrap()))
            .finish()
    }
}

#[cfg(feature = "debug-location")]
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
            .field("device", &self.device())
            .field("op", &self.op())
            .field(
                "inputs",
                &self.inputs().iter().map(|v| v.id()).collect::<Vec<_>>(),
            )
            .field("data", &format_args!("{}", data.as_deref().unwrap()))
            .field("location", &format_args!("{}", &self.0.location))
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

impl safetensors::View for Tensor {
    fn dtype(&self) -> safetensors::Dtype {
        self.0.dtype.safetensor_dtype()
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    fn data(&self) -> Cow<[u8]> {
        if !self.is_evaluated() {
            eval((self, true));
        }
        let data = self.0.data.borrow();
        let data = data.as_deref().unwrap();
        let bytes = data.as_bytes();
        assert_eq!(bytes.len(), self.data_len());
        bytes.into()
    }

    fn data_len(&self) -> usize {
        // number of elements * byte size of element
        self.0.shape.elem_count() * self.0.dtype.size_of_elem()
    }
}

impl<'a> safetensors::View for &'a Tensor {
    fn dtype(&self) -> safetensors::Dtype {
        self.0.dtype.safetensor_dtype()
    }

    fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    fn data(&self) -> Cow<[u8]> {
        if !self.is_evaluated() {
            eval((self, true));
        }
        let data = self.0.data.borrow();
        let data = data.as_deref().unwrap();
        let bytes = data.as_bytes();
        assert_eq!(bytes.len(), self.data_len());
        bytes.into()
    }

    fn data_len(&self) -> usize {
        // number of elements * byte size of element
        self.0.shape.elem_count() * self.0.dtype.size_of_elem()
    }
}

// custom drop implementation to avoid stack overflow in recursive drop
impl Drop for TensorImpl {
    fn drop(&mut self) {
        let mut inputs = self.inputs.borrow_mut();
        let inputs = inputs.drain(..);
        let mut tensors = Vec::new();
        for input in inputs {
            if let Ok(t) = Rc::try_unwrap(input.0) {
                tensors.push(t);
            }
        }
        while let Some(t) = tensors.pop() {
            let mut inputs = t.inputs.borrow_mut();
            let inputs = inputs.drain(..);
            for input in inputs {
                if let Ok(t) = Rc::try_unwrap(input.0) {
                    tensors.push(t);
                }
            }
        }
    }
}
