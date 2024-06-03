use crate::{Dim, Shape, Tensor};
use std::{
    fmt::Debug,
    ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

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
