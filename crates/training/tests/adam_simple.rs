use nexora_deeplearning::autograd::{Adam, Tensor, TensorOps};

#[test]
fn test_adam_manual_trace() {
    let target = 3.14159f32;
    let target_sq = target * target;

    // Test 1: Check mul backward
    let x = Tensor::from_slice(&[1.0f32], &[1]);
    x.set_requires_grad(true);
    let pred = x.clone().mul(&x.clone());
    pred.backward();
    let g_mul = x.grad();
    eprintln!("mul-only: x=1.0, pred=x², grad={:?} (expected Some([2.0]))", g_mul);
    x.zero_grad();

    // Test 2: Check sub backward
    let x2 = Tensor::from_slice(&[5.0f32], &[1]);
    x2.set_requires_grad(true);
    let const_t = Tensor::from_slice(&[3.0f32], &[1]);
    let diff = x2.sub(&const_t);
    diff.backward();
    let g_sub = x2.grad();
    eprintln!("sub-only: x=5.0, diff=x-3, grad={:?} (expected Some([1.0]))", g_sub);

    // Test 3: Check full chain
    let x3 = Tensor::from_slice(&[1.0f32], &[1]);
    x3.set_requires_grad(true);
    let pred3 = x3.clone().mul(&x3.clone()); // x²
    let t3 = Tensor::from_slice(&[9.0f32], &[1]);
    let diff3 = pred3.sub(&t3);              // x² - 9
    let loss3 = diff3.powf(2.0).mean();      // (x² - 9)²
    loss3.backward();
    let g_full = x3.grad();
    // d/dx (x² - 9)² = 2(x² - 9)*2x = 4x(x² - 9)
    // at x=1: 4*1*(1-9) = -32
    eprintln!("full chain: x=1.0, (x²-9)², grad={:?} (expected Some([-32.0]))", g_full);

    // Test 4: Same with π²
    let x4 = Tensor::from_slice(&[1.0f32], &[1]);
    x4.set_requires_grad(true);
    let pred4 = x4.clone().mul(&x4.clone());
    let t4 = Tensor::from_slice(&[target_sq], &[1]);
    let diff4 = pred4.sub(&t4);
    let loss4 = diff4.powf(2.0).mean();
    loss4.backward();
    let g_pi = x4.grad();
    // 4*1*(1-π²) = 4*(1-9.87) = -35.48
    eprintln!("pi chain: x=1.0, (x²-π²)², grad={:?} (expected Some([-35.48]))", g_pi);
}
