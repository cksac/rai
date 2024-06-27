use crate::{
    ty_kind, AsDevice, Error, GenericValue, GradMap, ParamMap, Result, Tensor, TensorMap, ValueSpec,
};
use std::{borrow::Cow, path::Path};

pub trait Module {
    type Input;
    type Output;
    fn forward(&self, x: &Self::Input) -> Self::Output;

    fn gather_params(&self, params: &mut TensorMap);

    fn params(&self) -> TensorMap {
        let mut params = TensorMap::new();
        self.gather_params(&mut params);
        params
    }

    fn update_params(&self, params: &mut TensorMap) -> Result<()>;

    fn gather_named_params(&self, prefix: &str, params: &mut ParamMap);

    fn named_params(&self, prefix: &str) -> ParamMap {
        let mut params = ParamMap::new();
        self.gather_named_params(prefix, &mut params);
        params
    }

    fn update_named_params(&self, prefix: &str, params: &mut ParamMap) -> Result<()>;

    fn to_safetensors<P: AsRef<Path>>(&self, filename: P)
    where
        Self: Sized,
    {
        let data = self.named_params("");
        safetensors::serialize_to_file(data, &None, filename.as_ref()).unwrap()
    }

    fn update_by_safetensors<P: AsRef<std::path::Path>>(
        &self,
        filenames: &[P],
        device: impl AsDevice,
    ) -> Result<()> {
        let mut st_tensors = ParamMap::new();
        let device = device.device();
        for filename in filenames {
            let data = std::fs::read(filename).unwrap();
            let st = safetensors::SafeTensors::deserialize(&data).unwrap();
            for (name, view) in st.tensors() {
                let t = Tensor::from_safetensor(&view, device);
                st_tensors.insert(name, t);
            }
        }
        self.update_named_params("", &mut st_tensors)
    }
}

impl<'a, T> Module for &'a T
where
    T: Module,
{
    type Input = T::Input;
    type Output = T::Output;

    #[inline]
    fn forward(&self, x: &Self::Input) -> Self::Output {
        (*self).forward(x)
    }

    #[inline]
    fn gather_params(&self, params: &mut TensorMap) {
        (*self).gather_params(params)
    }

    #[inline]
    fn update_params(&self, params: &mut TensorMap) -> Result<()> {
        (*self).update_params(params)
    }

    fn gather_named_params(&self, prefix: &str, params: &mut ParamMap) {
        (*self).gather_named_params(prefix, params)
    }

    fn update_named_params(&self, prefix: &str, params: &mut ParamMap) -> Result<()> {
        (*self).update_named_params(prefix, params)
    }
}

pub trait TrainableModule:
    Module + ValueSpec<Kind = ty_kind::Module, Tensors = TensorMap, Gradient = GradMap>
{
}

impl<'a, T> TrainableModule for &'a T where T: TrainableModule {}

impl<T> GenericValue<ty_kind::Module, TensorMap, GradMap> for T
where
    T: TrainableModule<Tensors = TensorMap, Gradient = GradMap>,
{
    #[inline]
    fn gv_tensors(&self) -> TensorMap {
        self.params()
    }

    #[inline]
    fn gv_grad(tensors: &TensorMap, grads: &GradMap) -> GradMap {
        tensors
            .keys()
            .map(|id| (*id, grads.get(*id).unwrap().clone()))
            .collect()
    }

    #[inline]
    fn gv_grad_map(tensors: &TensorMap, grad: GradMap, grads: &mut GradMap) {
        for id in tensors.keys() {
            grads.insert(*id, grad.get(*id).unwrap().clone());
        }
    }
}

pub trait NonTrainableModule:
    Module + ValueSpec<Kind = ty_kind::Module, Tensors = (), Gradient = ()>
{
}
impl<'a, T> NonTrainableModule for &'a T where T: NonTrainableModule {}

impl<T> GenericValue<ty_kind::Module, (), ()> for T
where
    T: NonTrainableModule<Tensors = (), Gradient = ()>,
{
    fn gv_tensors(&self) {}
    fn gv_grad(_: &(), _: &GradMap) {}
    fn gv_grad_map(_: &(), _: (), _: &mut GradMap) {}
}

pub trait ApplyModule<M>
where
    M: Module<Input = Self>,
{
    #[inline]
    fn apply(&self, module: M) -> M::Output {
        module.forward(self)
    }
}
impl<T, M> ApplyModule<M> for T where M: Module<Input = T> {}

pub trait WithParams {
    fn gather_by_id(&self, params: &mut TensorMap);
    fn update_by_id(&self, params: &mut TensorMap) -> Result<()>;
    fn gather_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str);
    fn update_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) -> Result<()>;
}

impl WithParams for Tensor {
    fn gather_by_id(&self, params: &mut TensorMap) {
        params.insert(self.id(), self.clone());
    }

    fn update_by_id(&self, params: &mut TensorMap) -> Result<()> {
        if let Some(t) = params.remove(self.id()) {
            // todo: check if can promote type
            let t = t.to_dtype(self).to_device(self);
            return self.replace_data(t);
        }
        // TODO: return error if tensor not found?
        Ok(())
    }

    fn gather_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) {
        let name = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name)
        };
        params.insert(name, self.clone());
    }

    fn update_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) -> Result<()> {
        let name: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        if let Some(t) = params.remove(name.as_ref()) {
            // todo: check if can promote type
            let t = t.to_dtype(self).to_device(self);
            self.replace_data(t)
        } else {
            Err(Error::ParamNotFound(name.into()))
        }
    }
}

impl<T> WithParams for Option<T>
where
    T: WithParams,
{
    fn gather_by_id(&self, params: &mut TensorMap) {
        if let Some(t) = self {
            t.gather_by_id(params);
        }
    }

    fn update_by_id(&self, params: &mut TensorMap) -> Result<()> {
        if let Some(t) = self {
            return t.update_by_id(params);
        }
        Ok(())
    }

    fn gather_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) {
        if let Some(t) = self {
            t.gather_by_name(params, prefix, name);
        }
    }

    fn update_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) -> Result<()> {
        if let Some(t) = self {
            return t.update_by_name(params, prefix, name);
        }
        Ok(())
    }
}

impl<T> WithParams for Vec<T>
where
    T: WithParams,
{
    fn gather_by_id(&self, params: &mut TensorMap) {
        for t in self {
            t.gather_by_id(params);
        }
    }

    fn update_by_id(&self, params: &mut TensorMap) -> Result<()> {
        for t in self {
            t.update_by_id(params)?;
        }
        Ok(())
    }

    fn gather_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) {
        for (i, t) in self.iter().enumerate() {
            let name = &format!("{}.{}", name, i);
            t.gather_by_name(params, prefix, name);
        }
    }

    fn update_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) -> Result<()> {
        for (i, t) in self.iter().enumerate() {
            let name = &format!("{}.{}", name, i);
            t.update_by_name(params, prefix, name)?;
        }
        Ok(())
    }
}

impl<T> WithParams for T
where
    T: Module,
{
    fn gather_by_id(&self, params: &mut TensorMap) {
        self.gather_params(params);
    }

    fn update_by_id(&self, params: &mut TensorMap) -> Result<()> {
        self.update_params(params)
    }

    fn gather_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) {
        let p: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        self.gather_named_params(&p, params)
    }

    fn update_by_name(&self, params: &mut ParamMap, prefix: &str, name: &str) -> Result<()> {
        let p: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        self.update_named_params(&p, params)
    }
}

pub trait ToApplyArg<O> {
    fn to_arg(self) -> O;
}

impl<'a, T> ToApplyArg<&'a T> for &'a T {
    fn to_arg(self) -> &'a T {
        self
    }
}

impl<'a, T> ToApplyArg<Option<&'a T>> for &'a Option<T> {
    fn to_arg(self) -> Option<&'a T> {
        self.as_ref()
    }
}

impl<'a, T: Copy> ToApplyArg<T> for &'a T {
    fn to_arg(self) -> T {
        *self
    }
}
