use burn::config::Config;
use burn::module::Module;
use burn::tensor::activation;
use burn::tensor::{backend::Backend, Tensor};
use super::grn::{GatedResidualNetwork, GatedResidualNetworkConfig};

#[derive(Module, Debug)]
pub struct VariableSelectionNetwork<B: Backend> {
    weight_network: GatedResidualNetwork<B>,
    variable_networks: Vec<GatedResidualNetwork<B>>,
}

impl<B: Backend> VariableSelectionNetwork<B> {
    pub fn forward<const D: usize>(
        &self,
        variables: Vec<Tensor<B, D>>,
        statics: Option<Tensor<B, D>>,
    ) -> (Tensor<B, D>, Tensor<B, D>) {
        let statics: Option<Tensor<B, D>> = match statics {
            None => None,
            Some(tensor) => {
                let dim_size = variables[0].dims()[1];
                Some(tensor.repeat(1, dim_size))
            }
        };

        let flattened = Tensor::cat(variables.clone(), D - 1);
        let weight = self.weight_network.forward(flattened, statics);

        let var_encodings: Vec<Tensor<B, D>> = self
            .variable_networks
            .iter()
            .zip(variables)
            .map(|(net, var)| {
                let enc = net.forward(var, None);
                enc
            })
            .collect();

        match D {
            2 => {
                let weight : Tensor<B, 3> = weight.unsqueeze_dim(1);
                let weight = activation::softmax(weight, 2);

                let var_encodings : Tensor<B, 3> = Tensor::stack(var_encodings, 2);
                let var_encodings = (var_encodings * weight.clone()).sum_dim(2);
                let var_encodings : Tensor<B, D> = var_encodings.squeeze(2);

                (var_encodings, weight.squeeze(1))
            },
            3 => {
                let weight : Tensor<B, 4> = weight.unsqueeze_dim(2);
                let weight = activation::softmax(weight, 3);

                let var_encodings : Tensor<B, 4> = Tensor::stack(var_encodings, 3);
                let var_encodings = (var_encodings * weight.clone()).sum_dim(3);
                let var_encodings : Tensor<B, D> = var_encodings.squeeze(3);

                (var_encodings, weight.squeeze(2))
            },
            _ => {
                panic!("Unsupported dimension")
            }
        }
    }
}

#[derive(Config, Debug)]
pub struct VariableSelectionNetworkConfig {
    d_hidden: usize,
    num_vars: usize,

    #[config(default = false)]
    add_static: bool,

    #[config(default = 0.0)]
    dropout: f64,
}

impl VariableSelectionNetworkConfig {
    pub fn init<B: Backend>(&self) -> VariableSelectionNetwork<B> {
        let d_static = if self.add_static { self.d_hidden } else { 0 };

        let weight_network: GatedResidualNetwork<B> =
            GatedResidualNetworkConfig::new(self.d_hidden)
                .with_d_input(Some(self.d_hidden * self.num_vars))
                .with_d_output(Some(self.num_vars))
                .with_d_static(d_static)
                .with_dropout(self.dropout)
                .init();

        let variable_networks: Vec<GatedResidualNetwork<B>> = (0..self.num_vars)
            .map(|_| {
                GatedResidualNetworkConfig::new(self.d_hidden)
                    .with_dropout(self.dropout)
                    .init()
            })
            .collect();

        VariableSelectionNetwork {
            weight_network,
            variable_networks,
        }
    }
}
