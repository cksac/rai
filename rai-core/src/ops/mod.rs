use crate::{shape::Dims, AsDevice, Dim, ElemType, Shape, Tensor};
use half::{bf16, f16};
use safetensors::tensor::TensorView;
use std::{any::Any, fmt::Debug};
use std::{
    f32::consts::PI,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
    slice::from_raw_parts,
};

mod full;
pub use full::*;

mod normal;
pub use normal::*;

mod random;
pub use random::*;

mod arange;
pub use arange::*;

mod from_array;
pub use from_array::*;

mod concatenate;
pub use concatenate::*;

mod add;
pub use add::*;

mod sub;
pub use sub::*;

mod mul;
pub use mul::*;

mod div;
pub use div::*;

mod matmul;
pub use matmul::*;

mod equal;
pub use equal::*;

mod not_equal;
pub use not_equal::*;

mod greater;
pub use greater::*;

mod greater_equal;
pub use greater_equal::*;

mod less;
pub use less::*;

mod less_equal;
pub use less_equal::*;

mod maximum;
pub use maximum::*;

mod minimum;
pub use minimum::*;

mod negative;
pub use negative::*;

mod sin;
pub use sin::*;

mod cos;
pub use cos::*;

mod square;
pub use square::*;

mod sqrt;
pub use sqrt::*;

mod rsqrt;
pub use rsqrt::*;

mod sign;
pub use sign::*;

mod abs;
pub use abs::*;

mod exp;
pub use exp::*;

mod log;
pub use log::*;

mod log2;
pub use log2::*;

mod log10;
pub use log10::*;

mod softmax;
pub use softmax::*;

mod log_softmax;
pub use log_softmax::*;

mod erf;
pub use erf::*;

mod tanh;
pub use tanh::*;

mod power_float;
pub use power_float::*;

mod broadcast;
pub use broadcast::*;

mod reshape;
pub use reshape::*;

mod transpose;
pub use transpose::*;

mod to_contiguous;
pub use to_contiguous::*;

mod to_device;
pub use to_device::*;

mod to_dtype;
pub use to_dtype::*;

mod permute;
pub use permute::*;

mod max_pool1d;
pub use max_pool1d::*;

mod max_pool2d;
pub use max_pool2d::*;

mod avg_pool1d;
pub use avg_pool1d::*;

mod avg_pool2d;
pub use avg_pool2d::*;

mod reduce_sum;
pub use reduce_sum::*;

mod reduce_max;
pub use reduce_max::*;

mod reduce_min;
pub use reduce_min::*;

mod argmax;
pub use argmax::*;

mod argmin;
pub use argmin::*;

mod gather;
pub use gather::*;

mod index_add;
pub use index_add::*;

mod scatter_add;
pub use scatter_add::*;

mod index_select;
pub use index_select::*;

mod narrow;
pub use narrow::*;

mod r#where;
pub use r#where::*;

mod conv1d;
pub use conv1d::*;

mod conv2d;
pub use conv2d::*;

mod conv_transpose1d;
pub use conv_transpose1d::*;

mod conv_transpose2d;
pub use conv_transpose2d::*;

mod upsample_nearest1d;
pub use upsample_nearest1d::*;

mod upsample_nearest2d;
pub use upsample_nearest2d::*;

mod flash_attention;
pub use flash_attention::*;

#[macro_export]
macro_rules! impl_std_ops_for_scalar {
    ($T:ty, $op:ident, $func:ident) => {
        impl std::ops::$op<Tensor> for $T {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: Tensor) -> Self::Output {
                let lhs = rhs.full_like::<$T>(self);
                $func(&lhs, &rhs)
            }
        }

        impl<'a> std::ops::$op<&'a Tensor> for $T {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: &'a Tensor) -> Self::Output {
                let lhs = rhs.full_like::<$T>(self);
                $func(&lhs, &rhs)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_std_ops {
    ($op:ident, $func:ident) => {
        impl std::ops::$op<Tensor> for Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(&self, &rhs)
            }
        }

        impl<'a> std::ops::$op<&'a Tensor> for Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: &'a Tensor) -> Tensor {
                $func(&self, rhs)
            }
        }

        impl<'a> std::ops::$op<Tensor> for &'a Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: Tensor) -> Tensor {
                $func(self, &rhs)
            }
        }

        impl<'a, 'b> std::ops::$op<&'b Tensor> for &'a Tensor {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: &'b Tensor) -> Tensor {
                $func(self, rhs)
            }
        }

        impl<T> std::ops::$op<T> for Tensor
        where
            T: crate::ElemType,
        {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like::<T>(rhs);
                $func(&self, &rhs)
            }
        }

        impl<'a, T> std::ops::$op<T> for &'a Tensor
        where
            T: crate::ElemType,
        {
            type Output = Tensor;

            #[track_caller]
            fn $func(self, rhs: T) -> Self::Output {
                let rhs = self.full_like::<T>(rhs);
                $func(self, &rhs)
            }
        }

        crate::impl_std_ops_for_scalar!(f32, $op, $func);
        crate::impl_std_ops_for_scalar!(f64, $op, $func);
        crate::impl_std_ops_for_scalar!(u8, $op, $func);
    };
}

#[macro_export]
macro_rules! broadcast_binary_op {
    ($(#[$meta:meta])* $primitive:ident, $func:ident) => {
        $(#[$meta])*
        #[track_caller]
        pub fn $func(lhs: &Tensor, rhs: &Tensor) -> Tensor {
            let device = lhs.device();
            let dtype = lhs.dtype();
            let (shape, lhs_b, rhs_b) = lhs.shape_broadcast_to(rhs).unwrap_or_else(|e| {
                panic!(
                    "{}({:?}, {:?}) with error {:?}",
                    stringify!($func),
                    lhs,
                    rhs,
                    e
                )
            });
            let inputs = match (lhs_b, rhs_b) {
                (false, false) => vec![lhs.clone(), rhs.clone()],
                (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&shape)],
                (true, false) => vec![lhs.broadcast_to_unchecked(&shape), rhs.clone()],
                (true, true) => vec![lhs.broadcast_to_unchecked(&shape), rhs.broadcast_to_unchecked(&shape)],
            };
            Tensor::new(device, dtype, shape, $primitive, inputs)
        }
    };
    ($(#[$meta:meta])* $primitive:ident, $func:ident, $out_ty:ident) => {
        $(#[$meta])*
        #[track_caller]
        pub fn $func(lhs: &Tensor, rhs: &Tensor) -> Tensor {
            let device = lhs.device();
            let dtype = $out_ty;
            let (shape, lhs_b, rhs_b) = lhs.shape_broadcast_to(rhs).unwrap_or_else(|e| {
                panic!(
                    "{}({:?}, {:?}) with error {:?}",
                    stringify!($func),
                    lhs,
                    rhs,
                    e
                )
            });
            let inputs = match (lhs_b, rhs_b) {
                (false, false) => vec![lhs.clone(), rhs.clone()],
                (false, true) => vec![lhs.clone(), rhs.broadcast_to_unchecked(&shape)],
                (true, false) => vec![lhs.broadcast_to_unchecked(&shape), rhs.clone()],
                (true, true) => vec![lhs.broadcast_to_unchecked(&shape), rhs.broadcast_to_unchecked(&shape)],
            };
            Tensor::new(device, dtype, shape, $primitive, inputs)
        }
    };
}

// Note: modified from candle candle_core::safetensors::convert_slice
/// Converts a byte slice to a typed slice.
///
/// # Arguments
///
/// * `data` - The byte slice to convert.
///
/// # Returns
///
/// A typed slice converted from the byte slice.
fn convert_slice<T: Clone>(data: &[u8]) -> Vec<T> {
    let size_in_bytes = std::mem::size_of::<T>();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize) % size_in_bytes == 0 {
        // SAFETY: This is safe because we just checked that this
        // was correctly aligned.
        let data: &[T] = unsafe { from_raw_parts(data.as_ptr() as *const T, elem_count) };
        data.to_vec()
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non-overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        c
    }
}

/// Creates a `Tensor` from a `safetensors::TensorView`.
///
/// # Arguments
///
/// * `view` - The `safetensors::TensorView` to create the `Tensor` from.
/// * `device` - The device to place the `Tensor` on.
///
/// # Returns
///
/// A `Tensor` created from the `safetensors::TensorView`.
#[track_caller]
pub fn from_safetensor(view: &TensorView, device: impl AsDevice) -> Tensor {
    let shape = view.shape();
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::BOOL => todo!(),
        safetensors::Dtype::U8 => {
            let data = convert_slice::<u8>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::I8 => todo!(),
        safetensors::Dtype::I16 => todo!(),
        safetensors::Dtype::U16 => todo!(),
        safetensors::Dtype::F16 => {
            let data = convert_slice::<f16>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::BF16 => {
            let data = convert_slice::<bf16>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::I32 => todo!(),
        safetensors::Dtype::U32 => {
            let data = convert_slice::<u32>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::F32 => {
            let data = convert_slice::<f32>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::F64 => {
            let data = convert_slice::<f64>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::I64 => {
            let data = convert_slice::<i64>(data);
            from_array(data, shape, device)
        }
        safetensors::Dtype::U64 => todo!(),
        _ => todo!(),
    }
}

pub trait ClampBound: Debug {
    fn bound(&self, input: &Tensor) -> Tensor;
}

impl<T: ElemType> ClampBound for T {
    fn bound(&self, input: &Tensor) -> Tensor {
        input.full_like(*self).to_dtype(input)
    }
}

impl ClampBound for Tensor {
    fn bound(&self, input: &Tensor) -> Tensor {
        self.to_dtype(input)
    }
}

impl ClampBound for &Tensor {
    fn bound(&self, input: &Tensor) -> Tensor {
        (*self).to_dtype(input)
    }
}

#[track_caller]
pub fn clamp(x: &Tensor, min: impl ClampBound, max: impl ClampBound) -> Tensor {
    let min = min.bound(x);
    let max = max.bound(x);
    x.maximum(min).minimum(max)
}

pub trait ReduceArgs: Debug {
    fn dims(&self) -> &impl Dims;
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> ReduceArgs for T
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        self
    }
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> VarArgs for T where T: Dims {}

impl<T> ReduceArgs for (T, bool)
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

impl<T> VarArgs for (T, bool) where T: Dims {}

impl<T> ReduceArgs for (T, usize)
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T> VarArgs for (T, usize)
where
    T: Dims,
{
    fn ddof(&self) -> usize {
        self.1
    }
}

impl<T> ReduceArgs for (T, bool, usize)
where
    T: Dims,
{
    fn dims(&self) -> &impl Dims {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

impl<T> VarArgs for (T, bool, usize)
where
    T: Dims,
{
    fn ddof(&self) -> usize {
        self.2
    }
}

#[track_caller]
pub fn mean<T: ReduceArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of(args.dims()) as f64;
    x.sum(args) / elem_count
}

pub trait VarArgs: ReduceArgs {
    fn ddof(&self) -> usize {
        0
    }
}

#[track_caller]
pub fn var<T: VarArgs>(x: &Tensor, args: T) -> Tensor {
    let elem_count = x.size_of(args.dims());
    let m = x.mean((args.dims(), args.keep_dim()));
    let s = (x - m).square().sum((args.dims(), args.keep_dim()));
    s / (elem_count - args.ddof()) as f32
}

#[track_caller]
pub fn relu(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like())
}

#[track_caller]
pub fn relu2(x: &Tensor) -> Tensor {
    x.maximum(x.zeros_like()).sqrt()
}

#[track_caller]
pub fn relu6(x: &Tensor) -> Tensor {
    x.clamp(0.0f32, 6.0f32)
}

#[track_caller]
pub fn gelu(x: &Tensor) -> Tensor {
    x * 0.5f32 * (1.0f32 + (x / 2.0f32.sqrt()).erf())
}

#[track_caller]
pub fn new_gelu(x: &Tensor) -> Tensor {
    0.5f32 * x * (1.0f32 + ((2.0f32 / PI).sqrt() * (x + 0.044715f32 * x.powf(3.0))).tanh())
}

#[track_caller]
pub fn silu(x: &Tensor) -> Tensor {
    x / (x.neg().exp() + 1.0f32)
}

pub trait FlattenArgs: Debug {
    fn start_dim(&self) -> impl Dim {
        0
    }
    fn end_dim(&self) -> impl Dim {
        -1
    }
}

impl FlattenArgs for usize {
    fn start_dim(&self) -> impl Dim {
        self
    }
}

impl FlattenArgs for isize {
    fn start_dim(&self) -> impl Dim {
        self
    }
}

impl FlattenArgs for i32 {
    fn start_dim(&self) -> impl Dim {
        self
    }
}

impl<D: Dim> FlattenArgs for (D, D) {
    fn start_dim(&self) -> impl Dim {
        &self.0
    }

    fn end_dim(&self) -> impl Dim {
        &self.1
    }
}

impl FlattenArgs for RangeFull {
    fn end_dim(&self) -> impl Dim {
        -1
    }
}

impl FlattenArgs for RangeTo<usize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeTo<isize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeTo<i32> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeToInclusive<usize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeToInclusive<isize> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeToInclusive<i32> {
    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeFrom<usize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }
}

impl FlattenArgs for RangeFrom<isize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }
}

impl FlattenArgs for RangeFrom<i32> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }
}

impl FlattenArgs for Range<usize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }

    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for Range<isize> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }

    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for Range<i32> {
    fn start_dim(&self) -> impl Dim {
        self.start
    }

    fn end_dim(&self) -> impl Dim {
        self.end
    }
}

impl FlattenArgs for RangeInclusive<usize> {
    fn start_dim(&self) -> impl Dim {
        self.start()
    }

    fn end_dim(&self) -> impl Dim {
        self.end()
    }
}

impl FlattenArgs for RangeInclusive<isize> {
    fn start_dim(&self) -> impl Dim {
        self.start()
    }

    fn end_dim(&self) -> impl Dim {
        self.end()
    }
}

impl FlattenArgs for RangeInclusive<i32> {
    fn start_dim(&self) -> impl Dim {
        self.start()
    }

    fn end_dim(&self) -> impl Dim {
        self.end()
    }
}

#[track_caller]
pub fn flatten<T: FlattenArgs>(x: &Tensor, args: T) -> Tensor {
    if x.ndim() == 0 {
        return x.reshape([1]);
    }
    let start_dim = x.dim(args.start_dim());
    let end_dim = x.dim(args.end_dim());
    if start_dim < end_dim {
        let mut dst_dim = x.shape_of(..start_dim);
        dst_dim.push(x.size_of(start_dim..=end_dim));
        if end_dim + 1 < x.ndim() {
            dst_dim.extend(x.shape_of(end_dim + 1..));
        }
        x.reshape(dst_dim)
    } else {
        x.clone()
    }
}

#[track_caller]
pub fn squeeze(x: &Tensor, dims: impl Dims) -> Tensor {
    let dims = x.dims(dims).to_vec();
    let mut out_shape = Vec::new();
    for (i, s) in x.shape().iter().enumerate() {
        if !dims.contains(&i) || *s != 1 {
            out_shape.push(*s);
        }
    }
    x.reshape(out_shape)
}

#[track_caller]
pub fn unsqueeze(x: &Tensor, d: impl Dim) -> Tensor {
    let is_negative = d.is_negative();
    let dim = x.dim(d);
    let mut shape = x.shape().to_vec();
    if is_negative {
        shape.insert(dim + 1, 1);
    } else {
        shape.insert(dim, 1);
    }
    x.reshape(shape)
}

#[track_caller]
pub fn chunk(x: &Tensor, chunks: usize, dim: impl Dim) -> Vec<Tensor> {
    let dim = x.dim(dim);
    let size = x.shape_at(dim);
    if size < chunks {
        (0..size).map(|i| x.narrow(dim, i, 1)).collect::<Vec<_>>()
    } else {
        let chunk_size = size / chunks;
        let cnt_additional = size % chunks;
        let mut tensors = vec![];
        let mut sum_chunk_size = 0;
        for i in 0..chunks {
            let chunk_size = if i < cnt_additional {
                chunk_size + 1
            } else {
                chunk_size
            };
            let tensor = x.narrow(dim, sum_chunk_size, chunk_size);
            tensors.push(tensor);
            sum_chunk_size += chunk_size
        }
        tensors
    }
}

pub trait ArgReduceArgs: Debug {
    fn dim(&self) -> &impl Dim;
    fn keep_dim(&self) -> bool {
        false
    }
}

impl<T: Dim> ArgReduceArgs for T {
    fn dim(&self) -> &impl Dim {
        self
    }
}

impl<T: Dim> ArgReduceArgs for (T, bool) {
    fn dim(&self) -> &impl Dim {
        &self.0
    }
    fn keep_dim(&self) -> bool {
        self.1
    }
}

pub trait ToPair<T>: Debug {
    fn to_pair(&self) -> (T, T);
}

impl ToPair<usize> for usize {
    fn to_pair(&self) -> (usize, usize) {
        (*self, *self)
    }
}

impl ToPair<usize> for [usize; 2] {
    fn to_pair(&self) -> (usize, usize) {
        (self[0], self[1])
    }
}

impl ToPair<usize> for (usize, usize) {
    fn to_pair(&self) -> (usize, usize) {
        *self
    }
}

#[track_caller]
pub fn dropout(input: &Tensor, p: f32) -> Tensor {
    assert!((0.0..1.0).contains(&p));
    let r = input.rand_like();
    let scale = 1.0 / (1.0 - p);
    let mask = r.ge(r.full_like(p)).to_dtype(r) * scale;
    input * mask
}

pub trait Op: Debug {
    fn clone_boxed(&self) -> Box<dyn Op>;
    fn dot_label(&self) -> String {
        format!("{:?}", self)
    }
    fn as_any(&self) -> &dyn Any;
    fn jvp(&self, output: &Tensor, primals: &[Tensor], tangents: &[Tensor]) -> Tensor;
    fn vjp(&self, output: &Tensor, primals: &[Tensor], cotangent: &Tensor) -> Vec<Tensor>;
}

impl<T> From<T> for Box<dyn Op>
where
    T: Clone + Op + 'static,
{
    fn from(t: T) -> Self {
        Box::new(t.clone())
    }
}

impl Clone for Box<dyn Op> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

impl<'a> From<&'a dyn Op> for Box<dyn Op> {
    fn from(t: &'a dyn Op) -> Self {
        t.clone_boxed()
    }
}

fn reduce_chooser_jvp_rule(g: &Tensor, ans: &Tensor, operand: &Tensor, dims: &[usize]) -> Tensor {
    let mut shape = operand.shape().to_vec();
    for dim in dims {
        shape[*dim] = 1;
    }
    let location_indicators = operand.eq(ans.reshape(shape)).to_dtype(g);
    let counts = location_indicators.sum(dims);
    (g * location_indicators).sum(dims) / counts
}
