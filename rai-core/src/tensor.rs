use crate::{
    eval,
    hlops::{self, ClampBound, FlattenArgs, VarArgs},
    nn::{ApplyModule, Module},
    ops::{
        self, ArangeArgs, ArgReduceArgs, AvgPool1dArgs, AvgPool2dArgs, MaxPool1dArgs,
        MaxPool2dArgs, ReduceArgs, ToPair,
    },
    utils::{self, dot_graph},
    AsDType, AsDevice, DType, Device, Dim, Dims, ElemType, FloatElemType, GradMap, Op, Shape, Type,
};
use safetensors::tensor::TensorView;
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

    #[inline]
    #[track_caller]
    pub fn full<T: ElemType>(val: T, shape: impl Shape, device: impl AsDevice) -> Tensor {
        ops::full::<T>(val, shape, device)
    }

    #[inline]
    #[track_caller]
    pub fn full_like<T: ElemType>(&self, val: T) -> Tensor {
        ops::full_like::<T>(self, val)
    }

    #[inline]
    #[track_caller]
    pub fn ones(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
        ops::ones(shape, dtype, device)
    }

    #[inline]
    #[track_caller]
    pub fn ones_like(&self) -> Tensor {
        ops::ones_like(self)
    }

    #[inline]
    #[track_caller]
    pub fn zeros(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
        ops::zeros(shape, dtype, device)
    }

    #[inline]
    #[track_caller]
    pub fn zeros_like(&self) -> Tensor {
        ops::zeros_like(self)
    }

    #[inline]
    #[track_caller]
    pub fn rand<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
        ops::rand(shape, dtype, device)
    }

    #[inline]
    #[track_caller]
    pub fn rand_with<T: ElemType>(
        from: T,
        to: T,
        shape: impl Shape,
        device: impl AsDevice,
    ) -> Tensor {
        ops::rand_with(from, to, shape, device)
    }

    #[inline]
    #[track_caller]
    pub fn rand_like(&self) -> Tensor {
        ops::rand_like(self)
    }

    #[inline]
    #[track_caller]
    pub fn randn<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
        ops::randn(shape, dtype, device)
    }

    #[inline]
    #[track_caller]
    pub fn randn_with<T: ElemType>(
        mean: T,
        std: T,
        shape: impl Shape,
        device: impl AsDevice,
    ) -> Tensor {
        ops::randn_with(mean, std, shape, device)
    }

    #[inline]
    #[track_caller]
    pub fn randn_like(&self) -> Tensor {
        ops::randn_like(self)
    }

    #[inline]
    #[track_caller]
    pub fn arange<D: Type, T: ArangeArgs<D>>(args: T, device: impl AsDevice) -> Tensor {
        ops::arange(args, device)
    }

    #[inline]
    #[track_caller]
    pub fn from_array<T: ElemType>(
        data: impl Into<Vec<T>> + Debug,
        shape: impl Shape,
        device: impl AsDevice,
    ) -> Tensor {
        ops::from_array(data, shape, device)
    }

    #[inline]
    #[track_caller]
    pub fn linspace<T: FloatElemType>(
        start: T,
        end: T,
        steps: usize,
        device: impl AsDevice,
    ) -> Tensor {
        ops::linspace(start, end, steps, device)
    }

    /// see [`ops::from_safetensor`](ops::from_safetensor)
    #[inline]
    #[track_caller]
    pub fn from_safetensor(view: &TensorView, device: impl AsDevice) -> Tensor {
        hlops::from_safetensor(view, device)
    }

    #[inline]
    #[track_caller]
    pub fn neg(&self) -> Tensor {
        ops::neg(self)
    }

    #[inline]
    #[track_caller]
    pub fn cat<T: AsRef<Tensor> + Debug>(tensors: &[T], dim: impl Dim) -> Tensor {
        ops::cat(tensors, dim)
    }

    #[inline]
    #[track_caller]
    pub fn matmul<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::matmul(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn eq<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::eq(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn ne<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::ne(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn gt<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::gt(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn ge<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::ge(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn lt<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::lt(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn le<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::le(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn maximum<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::maximum(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn minimum<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::minimum(self, rhs.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn t(&self) -> Tensor {
        ops::transpose(self, -2, -1)
    }

    #[inline]
    #[track_caller]
    pub fn transpose(&self, dim0: impl Dim, dim1: impl Dim) -> Tensor {
        ops::transpose(self, dim0, dim1)
    }

    #[inline]
    #[track_caller]
    pub fn broadcast_to(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_to(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn broadcast_to_unchecked(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_to_unchecked(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn broadcast_left(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_left(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn broadcast_right(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_right(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn reshape(&self, shape: impl Shape) -> Tensor {
        ops::reshape(self, shape)
    }

    #[inline]
    #[track_caller]
    pub fn add<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Add<T, Output = Tensor>,
    {
        std::ops::Add::add(self, rhs)
    }

    #[inline]
    #[track_caller]
    pub fn mul<T>(&self, rhs: T) -> Tensor
    where
        for<'a> &'a Self: std::ops::Mul<T, Output = Tensor>,
    {
        std::ops::Mul::mul(self, rhs)
    }

    #[inline]
    #[track_caller]
    pub fn sin(&self) -> Tensor {
        ops::sin(self)
    }

    #[inline]
    #[track_caller]
    pub fn cos(&self) -> Tensor {
        ops::cos(self)
    }

    #[inline]
    #[track_caller]
    pub fn tanh(&self) -> Tensor {
        ops::tanh(self)
    }

    #[inline]
    #[track_caller]
    pub fn square(&self) -> Tensor {
        ops::square(self)
    }

    #[inline]
    #[track_caller]
    pub fn powf(&self, exponent: f64) -> Tensor {
        ops::powf(self, exponent)
    }

    #[inline]
    #[track_caller]
    pub fn sqrt(&self) -> Tensor {
        ops::sqrt(self)
    }

    #[inline]
    #[track_caller]
    pub fn rsqrt(&self) -> Tensor {
        ops::rsqrt(self)
    }

    #[inline]
    #[track_caller]
    pub fn sign(&self) -> Tensor {
        ops::sign(self)
    }

    #[inline]
    #[track_caller]
    pub fn abs(&self) -> Tensor {
        ops::abs(self)
    }

    #[inline]
    #[track_caller]
    pub fn exp(&self) -> Tensor {
        ops::exp(self)
    }

    #[inline]
    #[track_caller]
    pub fn log(&self) -> Tensor {
        ops::log(self)
    }

    #[inline]
    #[track_caller]
    pub fn log2(&self) -> Tensor {
        ops::log2(self)
    }

    #[inline]
    #[track_caller]
    pub fn log10(&self) -> Tensor {
        ops::log10(self)
    }

    #[inline]
    #[track_caller]
    pub fn sum<T: ReduceArgs>(&self, args: T) -> Tensor {
        ops::sum(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn max<T: ReduceArgs>(&self, args: T) -> Tensor {
        ops::max(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn min<T: ReduceArgs>(&self, args: T) -> Tensor {
        ops::min(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn mean<T: ReduceArgs>(&self, args: T) -> Tensor {
        hlops::mean(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn var<T: VarArgs>(&self, args: T) -> Tensor {
        hlops::var(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn argmax<T: ArgReduceArgs>(&self, args: T) -> Tensor {
        ops::argmax(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn argmin<T: ArgReduceArgs>(&self, args: T) -> Tensor {
        ops::argmin(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn gather(&self, dim: impl Dim, index: impl AsRef<Tensor>) -> Tensor {
        ops::gather(self, dim, index.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn index_add(&self, dim: impl Dim, index: &Tensor, source: &Tensor) -> Tensor {
        ops::index_add(self, dim, index, source)
    }

    #[inline]
    #[track_caller]
    pub fn index_select(&self, dim: impl Dim, index: impl AsRef<Tensor>) -> Tensor {
        ops::index_select(self, dim, index.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn to_dtype(&self, dtype: impl AsDType) -> Tensor {
        ops::to_dtype(self, dtype)
    }

    #[inline]
    #[track_caller]
    pub fn to_device(&self, device: impl AsDevice) -> Tensor {
        ops::to_device(self, device)
    }

    #[inline]
    #[track_caller]
    pub fn softmax<D: Dim>(&self, d: D) -> Tensor {
        ops::softmax(self, d)
    }

    #[inline]
    #[track_caller]
    pub fn log_softmax<D: Dim>(&self, d: D) -> Tensor {
        ops::log_softmax(self, d)
    }

    #[inline]
    #[track_caller]
    pub fn erf(&self) -> Tensor {
        ops::erf(self)
    }

    #[inline]
    #[track_caller]
    pub fn clamp(&self, min: impl ClampBound, max: impl ClampBound) -> Tensor {
        hlops::clamp(self, min, max)
    }

    #[inline]
    #[track_caller]
    pub fn relu(&self) -> Tensor {
        hlops::relu(self)
    }

    #[inline]
    #[track_caller]
    pub fn relu2(&self) -> Tensor {
        hlops::relu2(self)
    }

    #[inline]
    #[track_caller]
    pub fn relu6(&self) -> Tensor {
        hlops::relu6(self)
    }

    #[inline]
    #[track_caller]
    pub fn gelu(&self) -> Tensor {
        hlops::gelu(self)
    }

    #[inline]
    #[track_caller]
    pub fn silu(&self) -> Tensor {
        hlops::silu(self)
    }

    #[inline]
    #[track_caller]
    pub fn new_gelu(&self) -> Tensor {
        hlops::new_gelu(self)
    }

    #[inline]
    #[track_caller]
    pub fn dropout(&self, p: f32) -> Tensor {
        hlops::dropout(self, p)
    }

    #[inline]
    #[track_caller]
    pub fn flatten<T: FlattenArgs>(&self, args: T) -> Tensor {
        hlops::flatten(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn to_contiguous(&self) -> Tensor {
        ops::to_contiguous(self)
    }

    #[inline]
    #[track_caller]
    pub fn squeeze(&self, d: impl Dims<Vec<usize>>) -> Tensor {
        hlops::squeeze(self, d)
    }

    #[inline]
    #[track_caller]
    pub fn unsqueeze(&self, d: impl Dim) -> Tensor {
        hlops::unsqueeze(self, d)
    }

    #[inline]
    #[track_caller]
    pub fn permute(&self, d: impl Dims<Vec<usize>>) -> Tensor {
        ops::permute(self, d)
    }

    #[inline]
    #[track_caller]
    pub fn narrow(&self, dim: impl Dim, start: usize, len: usize) -> Tensor {
        ops::narrow(self, dim, start, len)
    }

    #[inline]
    #[track_caller]
    pub fn chunk(&self, chunks: usize, dim: impl Dim) -> Vec<Tensor> {
        hlops::chunk(self, chunks, dim)
    }

    #[inline]
    #[track_caller]
    pub fn where_cond(&self, input: impl AsRef<Tensor>, other: impl AsRef<Tensor>) -> Tensor {
        ops::where_cond(self, input.as_ref(), other.as_ref())
    }

    #[inline]
    #[track_caller]
    pub fn conv1d(
        &self,
        kernel: impl AsRef<Tensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Tensor {
        ops::conv1d(self, kernel.as_ref(), padding, stride, dilation, groups)
    }

    #[inline]
    #[track_caller]
    pub fn conv_transpose1d(
        &self,
        kernel: impl AsRef<Tensor>,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Tensor {
        ops::conv_transpose1d(
            self,
            kernel.as_ref(),
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        )
    }

    #[inline]
    #[track_caller]
    pub fn conv2d(
        &self,
        kernel: impl AsRef<Tensor>,
        padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Tensor {
        ops::conv2d(self, kernel.as_ref(), padding, stride, dilation, groups)
    }

    #[inline]
    #[track_caller]
    pub fn conv_transpose2d(
        &self,
        kernel: impl AsRef<Tensor>,
        padding: [usize; 2],
        output_padding: [usize; 2],
        stride: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Tensor {
        ops::conv_transpose2d(
            self,
            kernel.as_ref(),
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        )
    }

    #[inline]
    #[track_caller]
    pub fn max_pool1d(&self, args: impl MaxPool1dArgs) -> Tensor {
        ops::max_pool1d(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn max_pool2d(&self, args: impl MaxPool2dArgs) -> Tensor {
        ops::max_pool2d(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn avg_pool1d(&self, args: impl AvgPool1dArgs) -> Tensor {
        ops::avg_pool1d(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn avg_pool2d(&self, args: impl AvgPool2dArgs) -> Tensor {
        ops::avg_pool2d(self, args)
    }

    #[inline]
    #[track_caller]
    pub fn upsample_nearest1d(&self, size: usize) -> Tensor {
        ops::upsample_nearest1d(self, size)
    }

    #[inline]
    #[track_caller]
    pub fn upsample_nearest2d(&self, size: impl ToPair<usize>) -> Tensor {
        ops::upsample_nearest2d(self, size)
    }

    #[inline]
    #[track_caller]
    pub fn scatter_add(
        &self,
        dim: impl Dim,
        index: impl AsRef<Tensor>,
        source: impl AsRef<Tensor>,
    ) -> Tensor {
        ops::scatter_add(self, dim, index.as_ref(), source.as_ref())
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
