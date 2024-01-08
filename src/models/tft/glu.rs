use burn::config::Config;
use burn::module::Module;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Clone, Debug)]
pub struct GatedLinearUnit {
    dim: i32,
    nonlinear: bool,
}

impl GatedLinearUnit {
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dim: usize = if self.dim < 0 {
            D - 1
        } else {
            self.dim as usize
        };

        let (value, gate) = {
            let mut chunks = x.chunk(2, dim);
            let value = chunks.remove(0);
            let gate = chunks.remove(0);
            (value, gate)
        };

        let value = if self.nonlinear { value.tanh() } else { value };

        let gate = activation::sigmoid(gate);

        gate * value
    }
}

#[derive(Config, Debug)]
pub struct GatedLinearUnitConfig {
    #[config(default = -1)]
    dim: i32,
    #[config(default = true)]
    nonlinear: bool,
}

impl GatedLinearUnitConfig {
    pub fn init(&self) -> GatedLinearUnit {
        GatedLinearUnit {
            dim: self.dim,
            nonlinear: self.nonlinear,
        }
    }
}
