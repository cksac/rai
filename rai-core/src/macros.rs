#[macro_export]
macro_rules! non_differentiable {
    ($kind:ident; $($path:tt)+) => {
        $crate::__non_differentiable!($kind; begin $($path)+);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __non_differentiable {
    // Invocation started with `<`, parse generics.
    ($kind:ident; begin < $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; generics () () $($rest)*);
    };

    // Invocation did not start with `<`.
    ($kind:ident; begin $first:tt $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; path () ($first) $($rest)*);
    };

    // End of generics.
    ($kind:ident; generics ($($generics:tt)*) () > $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; path ($($generics)*) () $($rest)*);
    };

    // Generics open bracket.
    ($kind:ident; generics ($($generics:tt)*) ($($brackets:tt)*) < $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; generics ($($generics)* <) ($($brackets)* <) $($rest)*);
    };

    // Generics close bracket.
    ($kind:ident; generics ($($generics:tt)*) (< $($brackets:tt)*) > $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; generics ($($generics)* >) ($($brackets)*) $($rest)*);
    };

    // Token inside of generics.
    ($kind:ident; generics ($($generics:tt)*) ($($brackets:tt)*) $first:tt $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; generics ($($generics)* $first) ($($brackets)*) $($rest)*);
    };

    // End with `where` clause.
    ($kind:ident; path ($($generics:tt)*) ($($path:tt)*) where $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; impl ($($generics)*) ($($path)*) ($($rest)*));
    };

    // End without `where` clause.
    ($kind:ident; path ($($generics:tt)*) ($($path:tt)*)) => {
        $crate::__non_differentiable!($kind; impl ($($generics)*) ($($path)*) ());
    };

    // Token inside of path.
    ($kind:ident; path ($($generics:tt)*) ($($path:tt)*) $first:tt $($rest:tt)*) => {
        $crate::__non_differentiable!($kind; path ($($generics)*) ($($path)* $first) $($rest)*);
    };

    // The impl.
    ($kind:ident; impl ($($generics:tt)*) ($($path:tt)*) ($($bound:tt)*)) => {
        impl<$($generics)*> $crate::ValueSpec for $($path)* where $($bound)* {
            type Kind = $kind;
            type Tensors = ();
            type Gradient = ();
        }

        impl<$($generics)*> $crate::GenericValue<$kind, (), ()> for $($path)* where $($bound)*
        {
            fn gv_tensors(&self) {}
            fn gv_grad(_: &(), _: &HashMap<usize, Tensor>) {}
            fn gv_grad_map(_: &(), _: (), _: &mut HashMap<usize, Tensor>) {}
        }
    };
}

#[macro_export]
macro_rules! non_trainable_module {
    ($M:ty) => {
        impl $crate::ValueSpec for $M {
            type Kind = $crate::ModuleValue;
            type Tensors = ();
            type Gradient = ();
        }

        impl $crate::NonTrainableModule for $M {}
    };
}

#[macro_export]
macro_rules! trainable_module {
    ($M:ty) => {
        impl $crate::ValueSpec for $M {
            type Kind = $crate::ModuleValue;
            type Tensors = HashMap<usize, Tensor>;
            type Gradient = HashMap<usize, Tensor>;
        }

        impl $crate::TrainableModule for $M {}
    };
}
