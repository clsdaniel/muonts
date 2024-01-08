use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

use crate::utils::split;

#[derive(Module, Debug)]
pub struct FeatureProjector<B: Backend> {
    feature_dims: Vec<usize>,
    projectors: Vec<Linear<B>>,
}

impl<B: Backend> FeatureProjector<B> {
    pub fn forward<const D: usize>(&self, features: Tensor<B, D>) -> Vec<Tensor<B, D>> {
        let feature_slices = if self.projectors.len() > 1 {
            split(features, self.feature_dims.clone(), -1)
        } else {
            vec![features]
        };

        self.projectors
            .iter()
            .zip(feature_slices)
            .map(|(proj, feat_slice)| proj.forward(feat_slice))
            .collect()
    }
}

#[derive(Config, Debug)]
pub struct FeatureProjectorConfig {
    feature_dims: Vec<usize>,
    embedding_dims: Vec<usize>,
}

impl FeatureProjectorConfig {
    pub fn init<B: Backend>(&self) -> FeatureProjector<B> {
        assert!(self.feature_dims.len() > 0);
        assert!(self.feature_dims.len() == self.embedding_dims.len());

        let projectors: Vec<Linear<B>> = self
            .feature_dims
            .iter()
            .zip(self.embedding_dims.iter())
            .map(|(c, d)| LinearConfig::new(*c, *d).init())
            .collect();

        FeatureProjector {
            feature_dims: self.feature_dims.clone(),
            projectors,
        }
    }
}
