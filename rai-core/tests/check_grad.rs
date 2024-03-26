use rai_core::{utils::check_grad, Cpu, Tensor, F64};

const EPS: f64 = 1e-4;

#[test]
fn check_add_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x + 2.0;
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sub_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x - 2.0;
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_mul_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x * 2.0;
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_div_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x / 2.0;
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_matmul_grad() {
    let device = Cpu;
    let y = &Tensor::rand([3, 2], F64, device);
    let func = |x: &Tensor| x.matmul(y);
    let x = &Tensor::rand([2, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_eq_grad() {
    let device = Cpu;
    let y = &Tensor::rand([2, 2], F64, device);
    let func = |x: &Tensor| x.eq(y).to_dtype(F64);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_gt_grad() {
    let device = Cpu;
    let y = &Tensor::rand([2, 2], F64, device);
    let func = |x: &Tensor| x.gt(y).to_dtype(F64);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_ge_grad() {
    let device = Cpu;
    let y = &Tensor::rand([2, 2], F64, device);
    let func = |x: &Tensor| x.ge(y).to_dtype(F64);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_lt_grad() {
    let device = Cpu;
    let y = &Tensor::rand([2, 2], F64, device);
    let func = |x: &Tensor| x.lt(y).to_dtype(F64);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_le_grad() {
    let device = Cpu;
    let y = &Tensor::rand([2, 2], F64, device);
    let func = |x: &Tensor| x.le(y).to_dtype(F64);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_maximum_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.maximum(x.full_like(0.5f64));
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_minimum_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.minimum(x.full_like(0.5f64));
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_cat_grad() {
    let device = Cpu;
    let y = &Tensor::rand([2, 2], F64, device);
    let func = |x: &Tensor| Tensor::cat(&[x, y], 0);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_gather_grad() {
    let device = Cpu;
    let y = &Tensor::from_array([1u8, 2], [2, 1], device);
    let func = |x: &Tensor| x.gather(-1, y);
    let x = &Tensor::rand([2, 4], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_index_add_grad() {
    let device = Cpu;
    let t = &Tensor::arange((0f64, 12f64), device).reshape([4, 3]);
    let idx = &Tensor::from_array([0u8, 1, 1], [3], device);
    let func = |x: &Tensor| x.index_add(1, idx, t);
    let x = &Tensor::rand([4, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_index_select_grad() {
    let device = Cpu;
    let idx = &Tensor::from_array([0u8, 1, 1], [3], device);
    let func = |x: &Tensor| x.index_select(1, idx);
    let x = &Tensor::rand([4, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_narrow_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.narrow(0, 0, 2);
    let x = &Tensor::rand([4, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_where_grad() {
    let device = Cpu;
    let pred = &Tensor::from_array([0u8, 1, 0, 1], [2, 2], device);
    let on_false = &Tensor::rand([2, 2], F64, device);
    let func = |x: &Tensor| pred.where_cond(x, on_false);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_scatter_add_grad() {
    let device = Cpu;
    let t = &Tensor::arange((0f64, 4f64), device).reshape([2, 2]);
    let idx = &Tensor::from_array([0u8, 0, 1, 1], [2, 2], device);
    let func = |x: &Tensor| x.scatter_add(1, idx, t);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sum_all_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum(..);
    let x = &Tensor::rand([2, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sum_all_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum((.., true));
    let x = &Tensor::rand([2, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sum_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum(-1);
    let x = &Tensor::rand([2, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sum_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sum((-1, true));
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_max_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.max((-1, true));
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_max_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.max(-1);
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_min_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.min((-1, true));
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_min_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.min(-1);
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_broadcast_to_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.broadcast_to([2, 3]);
    let x = &Tensor::rand([1, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_broadcast_left_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.broadcast_left([2, 3]);
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_broadcast_right_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.broadcast_right([2, 3]);
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_unsqueeze_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.unsqueeze(-1);
    let x = &Tensor::rand([2, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_reshape_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.reshape([3, 2]);
    let x = &Tensor::rand([2, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_transpose_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.transpose(0, 2);
    let x = &Tensor::rand([2, 3, 4], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_transpose2d_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.t();
    let x = &Tensor::rand([2, 3], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_permute_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.permute([2, 1, 0]);
    let x = &Tensor::rand([2, 3, 4], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_neg_grad() {
    let device = Cpu;
    let func = |x: &Tensor| -x;
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sin_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sin();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_cos_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.cos();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_square_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.square();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sqrt_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sqrt();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_rsqrt_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.rsqrt();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_sign_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.sign();
    let x = &Tensor::rand_with(-1.0f64, 1.0, [2, 2], device);
    check_grad(func, x, EPS);
}

#[test]
fn check_abs_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.abs();
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_exp_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.exp();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_log_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.log();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_log2_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.log2();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_log10_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.log10();
    let x = &Tensor::rand([1], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_softmax_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.softmax(-1);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_log_softmax_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.log_softmax(-1);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_erf_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.erf();
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_tanh_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.tanh();
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_powf_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.powf(2.0);
    let x = &Tensor::rand([2, 2], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_conv1d_grad() {
    let device = Cpu;
    let w = &Tensor::rand([2, 4, 3], F64, device);
    let x = &Tensor::rand([1, 4, 5], F64, device);
    let func = |t: &Tensor| t.conv1d(w, 0, 1, 1, 1);
    check_grad(func, x, EPS);
    let func = |t: &Tensor| x.conv1d(t, 0, 1, 1, 1);
    check_grad(func, w, EPS);
}

#[test]
fn check_conv2d_grad() {
    let device = Cpu;
    let w = &Tensor::rand([1, 2, 1, 1], F64, device);
    let x = &Tensor::rand([1, 2, 3, 3], F64, device);
    let func = |t: &Tensor| t.conv2d(w, [0, 0], [1, 1], [1, 1], 1);
    check_grad(func, x, EPS);
    let func = |t: &Tensor| x.conv2d(t, [0, 0], [1, 1], [1, 1], 1);
    check_grad(func, w, EPS);
}

#[test]
fn check_conv_transpose1d_grad() {
    let device = Cpu;
    let x = &Tensor::rand([1, 4, 5], F64, device);
    let w = &Tensor::rand([4, 2, 3], F64, device);
    let func = |t: &Tensor| t.conv_transpose1d(w, 0, 0, 1, 1, 1);
    check_grad(func, x, EPS);
    let func = |t: &Tensor| x.conv_transpose1d(t, 0, 0, 1, 1, 1);
    check_grad(func, w, EPS);
}

#[test]
fn check_conv_transpose2d_grad() {
    let device = Cpu;
    let x = &Tensor::rand([1, 4, 5, 5], F64, device);
    let w = &Tensor::rand([4, 2, 3, 3], F64, device);
    let func = |t: &Tensor| t.conv_transpose2d(w, [0, 0], [0, 0], [1, 1], [1, 1], 1);
    check_grad(func, x, EPS);
    let func = |t: &Tensor| x.conv_transpose2d(t, [0, 0], [0, 0], [1, 1], [1, 1], 1);
    check_grad(func, w, EPS);
}

#[test]
fn check_mean_all_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.mean(..);
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_mean_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.mean((-1, true));
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_var_all_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.var(..);
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}

#[test]
fn check_var_keep_dim_grad() {
    let device = Cpu;
    let func = |x: &Tensor| x.var((-1, true));
    let x = &Tensor::rand([2, 3, 5], F64, device);
    check_grad(func, x, EPS);
}
