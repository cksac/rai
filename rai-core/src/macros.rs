#[macro_export]
macro_rules! non_differentiable {
    ($($path:tt)+) => {
        $crate::__non_differentiable!(begin $($path)+);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __non_differentiable {
    // Invocation started with `<`, parse generics.
    (begin < $($rest:tt)*) => {
        $crate::__non_differentiable!(generics () () $($rest)*);
    };

    // Invocation did not start with `<`.
    (begin $first:tt $($rest:tt)*) => {
        $crate::__non_differentiable!(path () ($first) $($rest)*);
    };

    // End of generics.
    (generics ($($generics:tt)*) () > $($rest:tt)*) => {
        $crate::__non_differentiable!(path ($($generics)*) () $($rest)*);
    };

    // Generics open bracket.
    (generics ($($generics:tt)*) ($($brackets:tt)*) < $($rest:tt)*) => {
        $crate::__non_differentiable!(generics ($($generics)* <) ($($brackets)* <) $($rest)*);
    };

    // Generics close bracket.
    (generics ($($generics:tt)*) (< $($brackets:tt)*) > $($rest:tt)*) => {
        $crate::__non_differentiable!(generics ($($generics)* >) ($($brackets)*) $($rest)*);
    };

    // Token inside of generics.
    (generics ($($generics:tt)*) ($($brackets:tt)*) $first:tt $($rest:tt)*) => {
        $crate::__non_differentiable!(generics ($($generics)* $first) ($($brackets)*) $($rest)*);
    };

    // End with `where` clause.
    (path ($($generics:tt)*) ($($path:tt)*) where $($rest:tt)*) => {
        $crate::__non_differentiable!(impl ($($generics)*) ($($path)*) ($($rest)*));
    };

    // End without `where` clause.
    (path ($($generics:tt)*) ($($path:tt)*)) => {
        $crate::__non_differentiable!(impl ($($generics)*) ($($path)*) ());
    };

    // Token inside of path.
    (path ($($generics:tt)*) ($($path:tt)*) $first:tt $($rest:tt)*) => {
        $crate::__non_differentiable!(path ($($generics)*) ($($path)* $first) $($rest)*);
    };

    // The impl.
    (impl ($($generics:tt)*) ($($path:tt)*) ($($bound:tt)*)) => {
        impl<$($generics)*> $crate::Differentiable for $($path)* where $($bound)* {
            type Tensors = ();
            type Gradient = ();
            fn tensors(&self) -> Self::Tensors {}
            fn grad(_: &Self::Tensors, _: &HashMap<usize, Tensor>) -> Self::Gradient {}
            fn grad_map(_: &Self::Tensors, _: Self::Gradient, _: &mut HashMap<usize, Tensor>) {}
        }
    };
}

#[macro_export]
macro_rules! differentiable_module {
    ($m:ident) => {
        impl $crate::Differentiable for $m {
            type Tensors = std::collections::HashMap<usize, Tensor>;
            type Gradient = std::collections::HashMap<usize, Tensor>;

            fn tensors(&self) -> Self::Tensors {
                $crate::Module::params(self)
            }

            fn grad(
                tensors: &Self::Tensors,
                grad_map: &std::collections::HashMap<usize, Tensor>,
            ) -> Self::Gradient {
                tensors
                    .keys()
                    .map(|id| (*id, grad_map.get(id).unwrap().clone()))
                    .collect()
            }

            fn grad_map(
                tensors: &Self::Tensors,
                grad: Self::Gradient,
                out: &mut std::collections::HashMap<usize, Tensor>,
            ) {
                for id in tensors.keys() {
                    out.insert(*id, grad.get(id).unwrap().clone());
                }
            }
        }

        impl $crate::DifferentiableModule for $m {}
    };
}
