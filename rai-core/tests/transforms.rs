use rai_core::backend::Cpu;

use rai_core::{eval, grad, jvp, vjp, DType, Func, Tensor};

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

    let (outputs, jvps) = jvp(func, [a, b], [at, bt]);
    eval((&outputs, &jvps));

    println!("{}", outputs[0]);
    println!("{}", jvps[0]);
}

#[test]
fn test_vjp() {
    let backend = &Cpu;
    let a = Tensor::full(1.0, [2, 3], DType::F32, backend);
    let b = Tensor::full(1.0, [2, 3], DType::F32, backend);

    let (outputs, vjp_fn) = vjp(func, [a, b]);

    let t1 = Tensor::full(1.0, [2, 3], DType::F32, backend);
    let vjps_t1 = vjp_fn([t1]);

    let t2 = Tensor::full(2.0, [2, 3], DType::F32, backend);
    let vjps_t2 = vjp_fn([t2]);

    eval((&outputs, [&vjps_t1, &vjps_t2]));

    println!("{}", outputs[0]);
    println!("{}", vjps_t1[0]);
    println!("{}", vjps_t1[1]);
    println!("{}", vjps_t2[0]);
    println!("{}", vjps_t2[1]);
}

fn f(x: &Tensor, y: &Tensor) -> Tensor {
    x * x * 2.0 + y * 3.0 + 3.0
}

#[test]
fn test_grad() {
    let backend = &Cpu;
    let grad_func = grad(f);

    let a = Tensor::full(10.0, [1], DType::F32, backend);
    let b = Tensor::full(5.0, [1], DType::F32, backend);
    let grads = grad_func.apply([a, b]);
    eval(&grads);

    println!("{}", grads[0]);
    println!("{}", grads[1]);
}

// #[test]
// fn test_grad_grad() {
//     let backend = &Cpu;
//     let grad_func = grad(grad(f));

//     let a = Tensor::full(10.0, [1], DType::F32, backend);
//     let b = Tensor::full(5.0, [1], DType::F32, backend);
//     let grads = grad_func.apply([a, b]);
//     eval(&grads);

//     println!("{}", grads[0]);
//     println!("{}", grads[1]);
// }
