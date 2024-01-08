use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

use crate::modules::elu::{ELUConfig, ELU};

use super::glu::{GatedLinearUnit, GatedLinearUnitConfig};

#[derive(Module, Debug)]
pub struct GatedResidualNetwork<B: Backend> {
    skip_proj: Option<Linear<B>>,
    mlp_input_linear: Linear<B>,
    mlp_elu: ELU,
    mlp_dropout_linear: Linear<B>,
    mlp_dropout: Dropout,
    mlp_glu_linear: Linear<B>,
    mlp_glu: GatedLinearUnit,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> GatedResidualNetwork<B> {
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
        c: Option<Tensor<B, D>>,
    ) -> Tensor<B, D> {
        let skip = match &self.skip_proj {
            Some(proj) => proj.forward(x.clone()),
            None => x.clone(),
        };

        let x = match c {
            Some(val) => Tensor::cat(vec![x, val], D - 1),
            None => x,
        };

        let x = self.mlp_input_linear.forward(x);
        let x = self.mlp_elu.forward(x);
        let x = self.mlp_dropout_linear.forward(x);
        let x = self.mlp_dropout.forward(x);
        let x = self.mlp_glu_linear.forward(x);
        let x = self.mlp_glu.forward(x);

        let x = self.layer_norm.forward(x + skip);

        x
    }
}

#[derive(Config, Debug)]
pub struct GatedResidualNetworkConfig {
    d_hidden: usize,

    #[config(default = "None")]
    d_input: Option<usize>,

    #[config(default = "None")]
    d_output: Option<usize>,

    #[config(default = 0)]
    d_static: usize,

    #[config(default = 0.0)]
    dropout: f64,
}

impl GatedResidualNetworkConfig {
    pub fn init<B: Backend>(&self) -> GatedResidualNetwork<B> {
        let d_hidden = self.d_hidden;
        let d_static = self.d_static;

        let d_input = match self.d_input {
            Some(val) => val,
            None => d_hidden,
        };
        let d_output = match self.d_output {
            Some(val) => val,
            None => d_input,
        };

        let skip_proj = if d_input != d_output {
            Some(LinearConfig::new(d_input, d_output).init())
        } else {
            None
        };

        GatedResidualNetwork {
            skip_proj,
            mlp_input_linear: LinearConfig::new(d_input + d_static, d_hidden).init(),
            mlp_elu: ELUConfig::new().init(),
            mlp_dropout_linear: LinearConfig::new(d_hidden, d_hidden).init(),
            mlp_dropout: DropoutConfig::new(self.dropout).init(),
            mlp_glu_linear: LinearConfig::new(d_hidden, d_output * 2).init(),
            mlp_glu: GatedLinearUnitConfig::new().with_nonlinear(false).init(),
            layer_norm: LayerNormConfig::new(d_output).init(),
        }
    }
}
