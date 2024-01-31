use crate::{AsDevice, GenericValue, ModuleValue, Tensor, ValueSpec};
use std::{borrow::Cow, collections::HashMap, path::Path};

pub trait Module {
    type Input;
    type Output;
    fn forward(&self, x: &Self::Input) -> Self::Output;

    fn gather_params(&self, params: &mut HashMap<usize, Tensor>);

    fn params(&self) -> HashMap<usize, Tensor> {
        let mut params = HashMap::new();
        self.gather_params(&mut params);
        params
    }

    fn update_params(&self, params: &mut HashMap<usize, Tensor>);

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>);

    fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut params = HashMap::new();
        self.gather_named_params(prefix, &mut params);
        params
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>);

    fn to_safetensors<P: AsRef<Path>>(&self, filename: P)
    where
        Self: Sized,
    {
        let data = self.named_params("");
        safetensors::serialize_to_file(&data, &None, filename.as_ref()).unwrap()
    }

    fn update_by_safetensors<P: AsRef<std::path::Path>>(
        &self,
        filenames: &[P],
        device: impl AsDevice,
    ) {
        let mut st_tensors: HashMap<String, Tensor> = HashMap::new();
        let device = device.device();
        for filename in filenames {
            let data = std::fs::read(filename).unwrap();
            let st = safetensors::SafeTensors::deserialize(&data).unwrap();
            for (name, view) in st.tensors() {
                let t = Tensor::from_safetensor(&view, device);
                st_tensors.insert(name, t);
            }
        }
        self.update_named_params("", &mut st_tensors);
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
    fn gather_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).gather_params(params)
    }

    #[inline]
    fn update_params(&self, params: &mut HashMap<usize, Tensor>) {
        (*self).update_params(params)
    }

    fn gather_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        (*self).gather_named_params(prefix, params)
    }

    fn update_named_params(&self, prefix: &str, params: &mut HashMap<String, Tensor>) {
        (*self).update_named_params(prefix, params)
    }
}

pub trait TrainableModule:
    Module
    + ValueSpec<
        Kind = ModuleValue,
        Tensors = HashMap<usize, Tensor>,
        Gradient = HashMap<usize, Tensor>,
    >
{
}

impl<'a, T> TrainableModule for &'a T where T: TrainableModule {}

impl<T> GenericValue<ModuleValue, HashMap<usize, Tensor>, HashMap<usize, Tensor>> for T
where
    T: TrainableModule<Tensors = HashMap<usize, Tensor>, Gradient = HashMap<usize, Tensor>>,
{
    #[inline]
    fn gv_tensors(&self) -> HashMap<usize, Tensor> {
        self.params()
    }

    #[inline]
    fn gv_grad(
        tensors: &HashMap<usize, Tensor>,
        grad_map: &HashMap<usize, Tensor>,
    ) -> HashMap<usize, Tensor> {
        tensors
            .keys()
            .map(|id| (*id, grad_map.get(id).unwrap().clone()))
            .collect()
    }

    #[inline]
    fn gv_grad_map(
        tensors: &HashMap<usize, Tensor>,
        grad: HashMap<usize, Tensor>,
        out: &mut HashMap<usize, Tensor>,
    ) {
        for id in tensors.keys() {
            out.insert(*id, grad.get(id).unwrap().clone());
        }
    }
}

pub trait NonTrainableModule:
    Module + ValueSpec<Kind = ModuleValue, Tensors = (), Gradient = ()>
{
}
impl<'a, T> NonTrainableModule for &'a T where T: NonTrainableModule {}

impl<T> GenericValue<ModuleValue, (), ()> for T
where
    T: NonTrainableModule<Tensors = (), Gradient = ()>,
{
    fn gv_tensors(&self) {}
    fn gv_grad(_: &(), _: &HashMap<usize, Tensor>) {}
    fn gv_grad_map(_: &(), _: (), _: &mut HashMap<usize, Tensor>) {}
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
    fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>);
    fn update_by_id(&self, params: &mut HashMap<usize, Tensor>);

    fn gather_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str);
    fn update_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str);
}

impl WithParams for Tensor {
    fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        params.insert(self.id(), self.clone());
    }

    fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        if let Some(t) = params.remove(&self.id()) {
            // todo: check if can promote type
            let t = t.to_dtype(self).to_device(self);
            self.replace_data(t);
        }
    }

    fn gather_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        let name = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name)
        };
        params.insert(name, self.clone());
    }

    fn update_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        let name: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        if let Some(t) = params.remove(name.as_ref()) {
            // todo: check if can promote type
            let t = t.to_dtype(self).to_device(self);
            self.replace_data(t);
        } else {
            panic!("parameter {} not found", name);
        }
    }
}

impl<T> WithParams for Option<T>
where
    T: WithParams,
{
    fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        if let Some(t) = self {
            t.gather_by_id(params);
        }
    }

    fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        if let Some(t) = self {
            t.update_by_id(params);
        }
    }

    fn gather_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        if let Some(t) = self {
            t.gather_by_name(params, prefix, name);
        }
    }

    fn update_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        if let Some(t) = self {
            t.update_by_name(params, prefix, name);
        }
    }
}

impl<T> WithParams for Vec<T>
where
    T: WithParams,
{
    fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        for t in self {
            t.gather_by_id(params);
        }
    }

    fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        for t in self {
            t.update_by_id(params);
        }
    }

    fn gather_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        for (i, t) in self.iter().enumerate() {
            let name = &format!("{}.{}", name, i);
            t.gather_by_name(params, prefix, name);
        }
    }

    fn update_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        for (i, t) in self.iter().enumerate() {
            let name = &format!("{}.{}", name, i);
            t.update_by_name(params, prefix, name);
        }
    }
}

impl<T> WithParams for T
where
    T: Module,
{
    fn gather_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        self.gather_params(params);
    }

    fn update_by_id(&self, params: &mut HashMap<usize, Tensor>) {
        self.update_params(params);
    }

    fn gather_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
        let p: Cow<'_, str> = if prefix.is_empty() {
            name.into()
        } else {
            format!("{}.{}", prefix, name).into()
        };
        self.gather_named_params(&p, params)
    }

    fn update_by_name(&self, params: &mut HashMap<String, Tensor>, prefix: &str, name: &str) {
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
