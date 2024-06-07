use crate::{AsDevice, RaiResult, Tensor};
use half::{bf16, f16};
use safetensors::tensor::TensorView;
use std::slice::from_raw_parts;

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
pub fn from_safetensor(view: &TensorView, device: impl AsDevice) -> RaiResult<Tensor> {
    let shape = view.shape();
    let data = view.data();
    match view.dtype() {
        safetensors::Dtype::BOOL => todo!(),
        safetensors::Dtype::U8 => {
            let data = convert_slice::<u8>(data);
            Tensor::from_array(data, shape, device)
        }
        safetensors::Dtype::I8 => todo!(),
        safetensors::Dtype::I16 => todo!(),
        safetensors::Dtype::U16 => todo!(),
        safetensors::Dtype::F16 => {
            let data = convert_slice::<f16>(data);
            Tensor::from_array(data, shape, device)
        }
        safetensors::Dtype::BF16 => {
            let data = convert_slice::<bf16>(data);
            Tensor::from_array(data, shape, device)
        }
        safetensors::Dtype::I32 => todo!(),
        safetensors::Dtype::U32 => {
            let data = convert_slice::<u32>(data);
            Tensor::from_array(data, shape, device)
        }
        safetensors::Dtype::F32 => {
            let data = convert_slice::<f32>(data);
            Tensor::from_array(data, shape, device)
        }
        safetensors::Dtype::F64 => {
            let data = convert_slice::<f64>(data);
            Tensor::from_array(data, shape, device)
        }
        safetensors::Dtype::I64 => {
            let data = convert_slice::<i64>(data);
            Tensor::from_array(data, shape, device)
        }
        safetensors::Dtype::U64 => todo!(),
        _ => todo!(),
    }
}

impl Tensor {
    /// see [`hlops::from_safetensor`](hlops::from_safetensor)
    #[inline]
    #[track_caller]
    pub fn from_safetensor(view: &TensorView, device: impl AsDevice) -> RaiResult<Tensor> {
        from_safetensor(view, device)
    }
}
