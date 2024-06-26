use rai_core::{AsDevice, Cpu, Cuda};

#[test]
fn test_device() {
    let cpu = Cpu;
    let cpu_ref = &Cpu;
    let cuda0 = Cuda::new(0);
    let cuda1 = Cuda::new(1);
    assert_ne!(cpu.device(), cuda0.device());
    assert_ne!(cuda0.device(), cuda1.device());
    assert_eq!(cuda0.device(), cuda0.device());
    assert_eq!(cpu.device(), cpu_ref);
    assert_eq!(cpu.device(), cpu_ref.device());
    assert!(cuda0.device() != cpu_ref);
    assert!(cpu.device() == cpu_ref);
    assert!(cpu.device() == cpu_ref.device());
    assert!(cpu.into_boxed_device() == cpu_ref.into_boxed_device());
}
