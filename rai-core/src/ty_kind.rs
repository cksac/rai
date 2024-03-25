#[derive(Debug, Clone, Copy)]
pub struct Basic;

#[derive(Debug, Clone, Copy)]
pub struct Tuple<T>(T);

#[derive(Debug, Clone, Copy)]
pub struct Array<T>(T);

#[derive(Debug, Clone, Copy)]
pub struct Module;
