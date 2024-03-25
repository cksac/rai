use crate::{nn::Module, ty_kind, Func};

impl<I, O, F> Func<ty_kind::Basic, I, O> for F
where
    F: Fn(I) -> O,
{
    fn apply(&self, input: I) -> O {
        self(input)
    }
}

impl<I, O, F> Func<ty_kind::Module, I, O> for F
where
    F: Fn(I) -> O,
    I: Module,
{
    fn apply(&self, input: I) -> O {
        self(input)
    }
}

macro_rules! impl_tuple_arg_fn {
    ($($T:tt)*) => {
        paste::paste! {
            impl<$($T,)* OUT, FUNC> Func<ty_kind::Tuple<($($T,)*)>, ($($T,)*), OUT> for FUNC
            where
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

macro_rules! impl_array_arg_fn {
    ($S:expr; $($N:expr)*; $($T:tt)*) => {
        paste::paste! {
            impl<I, OUT, FUNC> Func<ty_kind::Array<[I; $S]>, [I; $S], OUT> for FUNC
            where
                FUNC: Fn($($T,)*) -> OUT,
            {
                fn apply(&self, input: [I; $S]) -> OUT {
                    let [$([<i $N>],)* ..] = input;
                    self($([<i $N>],)*)
                }
            }
        }
    };
}

impl_array_arg_fn!(1; 0; I);
impl_array_arg_fn!(2; 0 1; I I);
impl_array_arg_fn!(3; 0 1 2; I I I);
impl_array_arg_fn!(4; 0 1 2 3; I I I I);
impl_array_arg_fn!(5; 0 1 2 3 4; I I I I I);
impl_array_arg_fn!(6; 0 1 2 3 4 5; I I I I I I);
impl_array_arg_fn!(7; 0 1 2 3 4 5 6; I I I I I I I);
impl_array_arg_fn!(8; 0 1 2 3 4 5 6 7; I I I I I I I I);
impl_array_arg_fn!(9; 0 1 2 3 4 5 6 7 8; I I I I I I I I I);
impl_array_arg_fn!(10; 0 1 2 3 4 5 6 7 8 9; I I I I I I I I I I);
impl_array_arg_fn!(11; 0 1 2 3 4 5 6 7 8 9 10; I I I I I I I I I I I);
impl_array_arg_fn!(12; 0 1 2 3 4 5 6 7 8 9 10 11; I I I I I I I I I I I I);
