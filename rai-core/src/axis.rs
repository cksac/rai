use crate::Shape;

pub trait Axis {
    fn of_shape<T: Shape>(&self, shape: &T) -> usize;
}

impl Axis for usize {
    fn of_shape<T: Shape>(&self, shape: &T) -> usize {
        assert!(*self < shape.ndim());
        *self
    }
}

impl Axis for isize {
    fn of_shape<T: Shape>(&self, shape: &T) -> usize {
        let axis = if *self > 0 {
            *self as usize
        } else {
            self.checked_add_unsigned(shape.ndim()).unwrap() as usize
        };
        assert!(axis < shape.ndim());
        axis
    }
}

