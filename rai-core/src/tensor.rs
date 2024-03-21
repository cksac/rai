use crate::{
    eval,
    nn::{ApplyModule, Module},
    ops::{
        self, ArangeArgs, ArgReduceArgs, AvgPool1dArgs, AvgPool2dArgs, ClampBound, FlattenArgs,
        MaxPool1dArgs, MaxPool2dArgs, ReduceArgs, ToPair, VarArgs,
    },
    utils::{self, dot_graph},
    AsDType, AsDevice, DType, Device, Dim, Dims, ElemType, Primitive, Shape, Type,
};
use safetensors::tensor::TensorView;
use std::{
    any::Any,
    borrow::Cow,
    cell::{Ref, RefCell},
    collections::HashMap,
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    ops::Deref,
    path::Path,
    rc::Rc,
    sync::atomic,
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
    id: usize,
    device: Box<dyn Device>,
    dtype: Box<dyn DType>,
    shape: Vec<usize>,
    primitive: Box<dyn Primitive>,
    inputs: RefCell<Vec<Tensor>>,
    data: RefCell<Option<Box<dyn TensorLike>>>,
}

impl Tensor {
    pub fn new(
        device: impl AsDevice,
        dtype: impl AsDType,
        shape: impl Shape,
        primitive: impl Into<Box<dyn Primitive>>,
        inputs: impl Into<Vec<Tensor>>,
    ) -> Self {
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        let id = COUNTER.fetch_add(1, atomic::Ordering::Relaxed);
        let inner = TensorImpl {
            id,
            device: device.into_boxed_device(),
            dtype: dtype.into_boxed_dtype(),
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
    pub fn device(&self) -> &dyn Device {
        self.0.device.as_ref()
    }

    #[inline]
    pub fn dtype(&self) -> &dyn DType {
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
    pub fn full<T: ElemType>(val: T, shape: impl Shape, device: impl AsDevice) -> Tensor {
        ops::full::<T>(val, shape, device)
    }

    #[inline]
    pub fn full_like<T: ElemType>(&self, val: T) -> Tensor {
        ops::full_like::<T>(self, val)
    }

    #[inline]
    pub fn ones(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
        ops::ones(shape, dtype, device)
    }

    #[inline]
    pub fn ones_like(&self) -> Tensor {
        ops::ones_like(self)
    }

    #[inline]
    pub fn zeros(shape: impl Shape, dtype: impl AsDType, device: impl AsDevice) -> Tensor {
        ops::zeros(shape, dtype, device)
    }

    #[inline]
    pub fn zeros_like(&self) -> Tensor {
        ops::zeros_like(self)
    }

    #[inline]
    pub fn rand<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
        ops::rand(shape, dtype, device)
    }

    #[inline]
    pub fn rand_with<T: Type>(
        from: T::Repr,
        to: T::Repr,
        shape: impl Shape,
        dtype: T,
        device: impl AsDevice,
    ) -> Tensor {
        ops::rand_with(from, to, shape, dtype, device)
    }

    #[inline]
    pub fn rand_like(&self) -> Tensor {
        ops::rand_like(self)
    }

    #[inline]
    pub fn randn<T: Type>(shape: impl Shape, dtype: T, device: impl AsDevice) -> Tensor {
        ops::randn(shape, dtype, device)
    }

    #[inline]
    pub fn randn_with<T: Type>(
        mean: T::Repr,
        std: T::Repr,
        shape: impl Shape,
        dtype: T,
        device: impl AsDevice,
    ) -> Tensor {
        ops::randn_with(mean, std, shape, dtype, device)
    }

    #[inline]
    pub fn randn_like(&self) -> Tensor {
        ops::randn_like(self)
    }

    #[inline]
    pub fn arange<D: Type, T: ArangeArgs<D>>(args: T, device: impl AsDevice) -> Tensor
    where
        D::Repr:
            std::ops::Sub<D::Repr, Output = D::Repr> + std::ops::Div<D::Repr, Output = D::Repr>,
        D::Repr: Into<f64> + Copy,
    {
        ops::arange(args, device)
    }

    #[inline]
    pub fn from_array<T: ElemType>(
        data: impl Into<Vec<T>> + Debug,
        shape: impl Shape,
        device: impl AsDevice,
    ) -> Tensor {
        ops::from_array(data, shape, device)
    }

    /// see [`ops::from_safetensor`](ops::from_safetensor)
    #[inline]
    pub fn from_safetensor(view: &TensorView, device: impl AsDevice) -> Tensor {
        ops::from_safetensor(view, device)
    }

    #[inline]
    pub fn neg(&self) -> Tensor {
        ops::neg(self)
    }

    #[inline]
    pub fn cat<T: AsRef<Tensor> + Debug>(tensors: &[T], dim: impl Dim) -> Tensor {
        ops::cat(tensors, dim)
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
    pub fn minimum<T: AsRef<Tensor>>(&self, rhs: T) -> Tensor {
        ops::minimum(self, rhs.as_ref())
    }

    #[inline]
    pub fn t(&self) -> Tensor {
        ops::transpose(self, -2, -1)
    }

    #[inline]
    pub fn transpose(&self, dim0: impl Dim, dim1: impl Dim) -> Tensor {
        ops::transpose(self, dim0, dim1)
    }

    #[inline]
    pub fn broadcast_to(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_to(self, shape)
    }

    #[inline]
    pub fn broadcast_to_unchecked(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_to_unchecked(self, shape)
    }

    #[inline]
    pub fn broadcast_left(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_left(self, shape)
    }

    #[inline]
    pub fn broadcast_right(&self, shape: impl Shape) -> Tensor {
        ops::broadcast_right(self, shape)
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
    pub fn tanh(&self) -> Tensor {
        ops::tanh(self)
    }

    #[inline]
    pub fn square(&self) -> Tensor {
        ops::square(self)
    }

    #[inline]
    pub fn powf(&self, exponent: f64) -> Tensor {
        ops::powf(self, exponent)
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
    pub fn argmax<T: ArgReduceArgs>(&self, args: T) -> Tensor {
        ops::argmax(self, args)
    }

    #[inline]
    pub fn argmin<T: ArgReduceArgs>(&self, args: T) -> Tensor {
        ops::argmin(self, args)
    }

    #[inline]
    pub fn gather(&self, dim: impl Dim, index: impl AsRef<Tensor>) -> Tensor {
        ops::gather(self, dim, index.as_ref())
    }

    #[inline]
    pub fn index_add(&self, dim: impl Dim, index: &Tensor, source: &Tensor) -> Tensor {
        ops::index_add(self, dim, index, source)
    }

    #[inline]
    pub fn index_select(&self, dim: impl Dim, index: impl AsRef<Tensor>) -> Tensor {
        ops::index_select(self, dim, index.as_ref())
    }

    #[inline]
    pub fn to_dtype(&self, dtype: impl AsDType) -> Tensor {
        ops::to_dtype(self, dtype)
    }

    #[inline]
    pub fn to_device(&self, device: impl AsDevice) -> Tensor {
        ops::to_device(self, device)
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
    pub fn erf(&self) -> Tensor {
        ops::erf(self)
    }

    #[inline]
    pub fn clamp(&self, min: impl ClampBound, max: impl ClampBound) -> Tensor {
        ops::clamp(self, min, max)
    }

    #[inline]
    pub fn relu(&self) -> Tensor {
        ops::relu(self)
    }

    #[inline]
    pub fn relu2(&self) -> Tensor {
        ops::relu2(self)
    }

    #[inline]
    pub fn relu6(&self) -> Tensor {
        ops::relu6(self)
    }

    #[inline]
    pub fn gelu(&self) -> Tensor {
        ops::gelu(self)
    }

    #[inline]
    pub fn silu(&self) -> Tensor {
        ops::silu(self)
    }

    #[inline]
    pub fn new_gelu(&self) -> Tensor {
        ops::new_gelu(self)
    }

    #[inline]
    pub fn flatten<T: FlattenArgs>(&self, args: T) -> Tensor {
        ops::flatten(self, args)
    }

    #[inline]
    pub fn to_contiguous(&self) -> Tensor {
        ops::to_contiguous(self)
    }

    #[inline]
    pub fn squeeze(&self, d: impl Dims) -> Tensor {
        ops::squeeze(self, d)
    }

    #[inline]
    pub fn unsqueeze(&self, d: impl Dim) -> Tensor {
        ops::unsqueeze(self, d)
    }

    #[inline]
    pub fn permute(&self, d: impl Dims) -> Tensor {
        ops::permute(self, d)
    }

    #[inline]
    pub fn narrow(&self, dim: impl Dim, start: usize, len: usize) -> Tensor {
        ops::narrow(self, dim, start, len)
    }

    #[inline]
    pub fn chunk(&self, chunks: usize, dim: impl Dim) -> Vec<Tensor> {
        ops::chunk(self, chunks, dim)
    }

    #[inline]
    pub fn where_cond(&self, input: impl AsRef<Tensor>, other: impl AsRef<Tensor>) -> Tensor {
        ops::where_cond(self, input.as_ref(), other.as_ref())
    }

    #[inline]
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
    pub fn max_pool1d(&self, args: impl MaxPool1dArgs) -> Tensor {
        ops::max_pool1d(self, args)
    }

    #[inline]
    pub fn max_pool2d(&self, args: impl MaxPool2dArgs) -> Tensor {
        ops::max_pool2d(self, args)
    }

    #[inline]
    pub fn avg_pool1d(&self, args: impl AvgPool1dArgs) -> Tensor {
        ops::avg_pool1d(self, args)
    }

    #[inline]
    pub fn avg_pool2d(&self, args: impl AvgPool2dArgs) -> Tensor {
        ops::avg_pool2d(self, args)
    }

    #[inline]
    pub fn upsample_nearest1d(&self, size: usize) -> Tensor {
        ops::upsample_nearest1d(self, size)
    }

    #[inline]
    pub fn upsample_nearest2d(&self, size: impl ToPair<usize>) -> Tensor {
        ops::upsample_nearest2d(self, size)
    }

    #[inline]
    pub fn scatter_add(
        &self,
        dim: impl Dim,
        index: impl AsRef<Tensor>,
        source: impl AsRef<Tensor>,
    ) -> Tensor {
        ops::scatter_add(self, dim, index.as_ref(), source.as_ref())
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

    pub fn vjp(&self, cotangent: &Tensor, grads_sum: &mut HashMap<usize, Tensor>) {
        let primals = &*self.inputs();
        if primals.is_empty() {
            return;
        }
        let cotangents = self.primitive().vjp(self, primals, cotangent);
        for (primal, cotan) in primals.iter().zip(cotangents.iter()) {
            let id = primal.id();
            if let Some(sum) = grads_sum.get(&id) {
                grads_sum.insert(id, sum + cotan);
            }
            primal.vjp(cotan, grads_sum);
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
            "{:?} not align with rhs shape: {:?}\n{}",
            self,
            rhs.shape(),
            dot_graph([self, &rhs])
        );
        assert!(self.dtype() == rhs.dtype(), "dtype must be equal");
        assert!(
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
            self.shape_eq(&data.shape()),
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

    #[inline]
    pub fn to_safetensors(&self, name: impl Into<String>, filename: impl AsRef<Path>) {
        let data = HashMap::from([(name.into(), self.clone())]);
        safetensors::serialize_to_file(&data, &None, filename.as_ref()).unwrap()
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
            .field("device", &self.device())
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
            .field("device", &self.device())
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
        self.0.shape.size() * self.0.dtype.size_of_elem()
    }
}
