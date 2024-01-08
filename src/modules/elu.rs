use burn::config::Config;
use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug, Clone)]
pub struct ELU {
    alpha: f32,
}

impl ELU {
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let mask = x.clone().lower_equal_elem(0.0);
        let value = (x.clone().exp() - 1) * self.alpha;

        x.mask_where(mask, value)
    }
}

#[derive(Config, Debug)]
pub struct ELUConfig {
    #[config(default = 1.0)]
    alpha: f32,
}

impl ELUConfig {
    pub fn init(&self) -> ELU {
        ELU { alpha: self.alpha }
    }
}
