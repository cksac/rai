use crate::{Differentiable, Func};

macro_rules! impl_tuple_arg_fn {
    ($($T:tt)*) => {
        paste::paste! {
            impl<$($T,)* OUT, FUNC> Func<($($T,)*), OUT> for FUNC
            where
                $($T: Differentiable,)*
                OUT: Differentiable,
                FUNC: Fn($($T,)*) -> OUT,
            {
                fn apply(&self, input: ($($T,)*)) -> OUT {
                    let ($([<$T:lower 1>],)*) = input;
                    self($([<$T:lower 1>],)*)
                }
            }
        }
    };
}

impl_tuple_arg_fn!(A);
impl_tuple_arg_fn!(A B);
impl_tuple_arg_fn!(A B C);
impl_tuple_arg_fn!(A B C D);
impl_tuple_arg_fn!(A B C D E);
impl_tuple_arg_fn!(A B C D E F);
impl_tuple_arg_fn!(A B C D E F G);
impl_tuple_arg_fn!(A B C D E F G H);
impl_tuple_arg_fn!(A B C D E F G H I);
impl_tuple_arg_fn!(A B C D E F G H I J);
impl_tuple_arg_fn!(A B C D E F G H I J K);
impl_tuple_arg_fn!(A B C D E F G H I J K L);

// macro_rules! impl_array_arg_fn {
//     ($S:expr; $($T:tt)*) => {
//         paste::paste! {
//             impl<I, OUT, FUNC> Func<[I; $S], OUT> for FUNC
//             where
//                 I: Differentiable,
//                 OUT: Differentiable,
//                 FUNC: Fn($($T,)*) -> OUT,
//             {
//                 fn apply(&self, input: [I; $S]) -> OUT {
//                     let [$([<$T:lower 1>],)* ..] = input;
//                     self($([<$T:lower 1>],)*)
//                 }
//             }
//         }
//     };
// }

//impl_array_arg_fn!(1; I);
// impl_array_arg_fn!(2; I I);
// impl_array_arg_fn!(3; I I I);
// impl_array_arg_fn!(4; I I I I);
// impl_array_arg_fn!(5; I I I I I);
// impl_array_arg_fn!(6; I I I I I I);
// impl_array_arg_fn!(7; I I I I I I I);
// impl_array_arg_fn!(8; I I I I I I I I);
// impl_array_arg_fn!(9; I I I I I I I I I);
// impl_array_arg_fn!(10; I I I I I I I I I I);
// impl_array_arg_fn!(11 I I I I I I I I I I I);
// impl_array_arg_fn!(12; I I I I I I I I I I I I);

impl<I, O, F> Func<[I; 1], O> for F
where
    F: Fn(I) -> O,
    I: Differentiable,
    O: Differentiable,
{
    fn apply(&self, input: [I; 1]) -> O {
        let [a, ..] = input;
        self(a)
    }
}
