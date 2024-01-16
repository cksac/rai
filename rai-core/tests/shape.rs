use rai_core::Shape;

#[test]
fn test_dim() {
    let s = [0, 1, 2, 3, 4, 5];
    let ndim = s.ndim();

    assert!(s.dim(0) == 0);
    assert!(s.dim(1) == 1);
    assert!(s.dim(2) == 2);
    assert!(s.dim(-2) == ndim - 3);
    assert!(s.dim(-1) == ndim - 2);
    assert!(s.dim(..) == ndim - 1);
}

#[test]
fn test_dims() {
    let s = [0, 1, 2, 3, 4, 5];

    assert_eq!(s.dims(0..0), (&[] as &[usize]));
    assert_eq!(s.dims(0..1), &[0usize]);
    assert_eq!(s.dims(2..4), &[2usize, 3]);

    assert_eq!(s.dims(4..2), &[4usize, 3]);
    assert_eq!(s.dims(4..0), &[4usize, 3, 2, 1]);

    assert_eq!(s.dims(0..=0), &[0usize]);
    assert_eq!(s.dims(0..=1), &[0usize, 1]);
    assert_eq!(s.dims(2..=4), &[2usize, 3, 4]);

    assert_eq!(s.dims(4..=2), &[4usize, 3, 2]);
    assert_eq!(s.dims(4..=0), &[4usize, 3, 2, 1, 0]);

    assert_eq!(s.dims(..0), (&[] as &[usize]));
    assert_eq!(s.dims(..1), &[0usize]);
    assert_eq!(s.dims(..2), &[0usize, 1]);

    assert_eq!(s.dims(..=0), &[0usize]);
    assert_eq!(s.dims(..=1), &[0usize, 1]);
    assert_eq!(s.dims(..=2), &[0usize, 1, 2]);

    assert_eq!(s.dims(0..), &[0usize, 1, 2, 3, 4, 5]);
    assert_eq!(s.dims(1..), &[1, 2, 3, 4, 5]);
    assert_eq!(s.dims(2..), &[2usize, 3, 4, 5]);

    assert_eq!(s.dims(..), &[0usize, 1, 2, 3, 4, 5]);

    assert_eq!(s.dims(-1..), &[4usize, 5]);
    assert_eq!(s.dims(-2..), &[3usize, 4, 5]);
    assert_eq!(s.dims(-5..), &[0usize, 1, 2, 3, 4, 5]);

    assert_eq!(s.dims(-1..0), &[4usize, 3, 2, 1]);
    assert_eq!(s.dims(-4..-1), &[1usize, 2, 3]);

    assert_eq!(s.dims(-1..=0), &[4usize, 3, 2, 1, 0]);
    assert_eq!(s.dims(-4..=-1), &[1usize, 2, 3, 4]);
}

#[test]
fn test_shape() {
    let s = [1, 2, 3, 4, 5, 6];
    assert_eq!(s.shape_at(..), 6);
}
