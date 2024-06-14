extern crate proc_macro;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::ToTokens;
use syn::{spanned::Spanned, DeriveInput, Ident, Path, Type};

#[derive(Debug, deluxe::ExtractAttributes)]
#[deluxe(attributes(module))]
#[deluxe(default)]
struct ContainerOpts {
    #[deluxe(rename = input)]
    input_ty: Option<Type>,

    #[deluxe(rename = output)]
    output_ty: Option<Type>,

    #[deluxe(rename = crate)]
    crate_root: Path,

    trainable: bool,
}
impl Default for ContainerOpts {
    fn default() -> Self {
        Self {
            input_ty: None,
            output_ty: None,
            crate_root: syn::parse_quote!(rai),
            trainable: true,
        }
    }
}

#[derive(Debug, deluxe::ParseAttributes)]
#[deluxe(attributes(param))]
struct FieldOpts<'t> {
    #[deluxe(container)]
    field: &'t syn::Field,
    #[deluxe(default)]
    rename: Option<String>,
    #[deluxe(default)]
    skip: bool,
}

#[proc_macro_derive(Module, attributes(module, param))]
pub fn module(item: TokenStream) -> TokenStream {
    let mut input: DeriveInput = syn::parse(item).expect("syn::parse ok");

    let errors = deluxe::Errors::new();
    let ContainerOpts {
        input_ty,
        output_ty,
        crate_root,
        trainable,
    } = deluxe::extract_attributes_optional(&mut input, &errors);

    let mut field_opts: Vec<FieldOpts> = Vec::new();
    let mut is_unit_struct = false;
    if let syn::Data::Struct(s) = &mut input.data {
        match &mut s.fields {
            syn::Fields::Named(fields) => {
                for field in fields.named.iter_mut() {
                    match deluxe::parse_attributes(field) {
                        Ok(f_opts) => field_opts.push(f_opts),
                        Err(e) => errors.push_syn(e),
                    }
                }
            }
            syn::Fields::Unit => is_unit_struct = true,
            syn::Fields::Unnamed(_) => errors.push(Span::call_site(), "tuple is not supported"),
        }
    }
    if !errors.is_empty() {
        return errors.into_token_stream().into();
    }

    let receiver_name = &input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let input_ty = input_ty.unwrap_or_else(|| {
        syn::parse_quote! {
            ::#crate_root::Tensor
        }
    });
    let output_ty = output_ty.unwrap_or_else(|| {
        syn::parse_quote! {
            ::#crate_root::Tensor
        }
    });

    let call_fwd = match &input_ty {
        Type::Path(_) | Type::Array(_) => {
            quote::quote! {
                self.fwd(input)
            }
        }
        Type::Tuple(tuple) => {
            let args: Vec<_> = tuple
                .elems
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let arg = Ident::new(&format!("a{i}"), t.span());
                    quote::quote! {
                        #arg
                    }
                })
                .collect();

            quote::quote! {
                let (#(#args,)*) = input;
                self.fwd(#(::#crate_root::nn::ToApplyArg::to_arg(#args),)*)
            }
        }
        _ => panic!("unsupported module input type"),
    };

    let module_impls = if is_unit_struct || !trainable {
        quote::quote! {
            impl #impl_generics ::#crate_root::nn::Module for #receiver_name #type_generics #where_clause {
                type Input = #input_ty;
                type Output = #output_ty;

                #[inline]
                fn forward(&self, input: &Self::Input) -> Self::Output {
                    #call_fwd
                }
                fn gather_params(&self, params: &mut ::#crate_root::TensorMap) {}
                fn update_params(&self, params: &mut ::#crate_root::TensorMap) -> #crate_root::Result<()> {
                    Ok(())
                }
                fn gather_named_params(&self, prefix: &str, params: &mut ::#crate_root::ParamMap) {}
                fn update_named_params(&self, prefix: &str, params: &mut ::#crate_root::ParamMap) -> #crate_root::Result<()> {
                    Ok(())
                }
            }

            impl #impl_generics ::#crate_root::ValueSpec for #receiver_name #type_generics #where_clause {
                type Kind = ::#crate_root::ty_kind::Module;
                type Tensors = ();
                type Gradient = ();
            }

            impl #impl_generics ::#crate_root::nn::NonTrainableModule for #receiver_name #type_generics #where_clause {}
        }
    } else {
        let update_params: Vec<_> = field_opts
            .iter()
            .filter(|f| !f.skip)
            .map(|f| {
                let field_name = f.field.ident.as_ref().unwrap();
                quote::quote! {
                    ::#crate_root::nn::WithParams::update_by_id(&self.#field_name, params)?;
                }
            })
            .collect();

        let gather_params: Vec<_> = field_opts
            .iter()
            .filter(|f| !f.skip)
            .map(|f| {
                let field_name = f.field.ident.as_ref().unwrap();
                quote::quote! {
                    ::#crate_root::nn::WithParams::gather_by_id(&self.#field_name, params);
                }
            })
            .collect();

        let update_named_params: Vec<_> = field_opts
            .iter()
            .filter(|f| !f.skip)
            .map(|f| {
                let field_name = f.field.ident.as_ref().unwrap();
                let f_name = field_name.to_string();
                let param_name = f.rename.as_ref().unwrap_or(&f_name);
                quote::quote! {
                    ::#crate_root::nn::WithParams::update_by_name(&self.#field_name, params, prefix, #param_name)?;
                }
            })
            .collect();

        let gather_named_params: Vec<_> = field_opts
            .iter()
            .filter(|f| !f.skip)
            .map(|f| {
                let field_name = f.field.ident.as_ref().unwrap();
                let f_name = field_name.to_string();
                let param_name = f.rename.as_ref().unwrap_or(&f_name);
                quote::quote! {
                    ::#crate_root::nn::WithParams::gather_by_name(&self.#field_name, params, prefix, #param_name);
                }
            })
            .collect();

        quote::quote! {
            impl #impl_generics ::#crate_root::nn::Module for #receiver_name #type_generics #where_clause {
                type Input = #input_ty;
                type Output = #output_ty;

                #[inline]
                fn forward(&self, input: &Self::Input) -> Self::Output {
                    #call_fwd
                }

                fn gather_params(&self, params: &mut ::#crate_root::TensorMap) {
                    #(#gather_params)*
                }

                fn update_params(&self, params: &mut ::#crate_root::TensorMap) -> #crate_root::Result<()> {
                    #(#update_params)*
                    Ok(())
                }

                fn gather_named_params(&self, prefix: &str, params: &mut ::#crate_root::ParamMap) {
                    #(#gather_named_params)*
                }

                fn update_named_params(&self, prefix: &str, params: &mut ::#crate_root::ParamMap) -> #crate_root::Result<()> {
                    #(#update_named_params)*
                    Ok(())
                }
            }

            impl #impl_generics ::#crate_root::ValueSpec for #receiver_name #type_generics #where_clause {
                type Kind = ::#crate_root::ty_kind::Module;
                type Tensors = ::#crate_root::TensorMap;
                type Gradient = ::#crate_root::GradMap;
            }

            impl #impl_generics ::#crate_root::nn::TrainableModule for #receiver_name #type_generics #where_clause {}
        }
    };

    module_impls.into()
}
