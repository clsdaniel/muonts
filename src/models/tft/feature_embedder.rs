use burn::config::Config;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::Int;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module, Debug)]
pub struct FeatureEmbedder<B: Backend> {
    embedders: Vec<Embedding<B>>,
}

impl<B: Backend> FeatureEmbedder<B> {
    pub fn forward<const D: usize>(&self, features: Tensor<B, D, Int>) -> Vec<Tensor<B, 3>> {
        let cat_feat_slices = if self.embedders.len() > 1 {
            features.chunk(self.embedders.len(), D - 1)
        } else {
            vec![features]
        };

        self.embedders
            .iter()
            .zip(cat_feat_slices)
            .map(|(emb, feat)| {
                //info!("Feat before {}", feat);
                let feat : Tensor<B, 2, Int> = if D == 3 {
                    feat.unsqueeze_dim(2)
                } else {
                    feat.unsqueeze()
                };
                //info!("Feat after {}", feat);
                emb.forward(feat)
            })
            .collect()
        // Output is [batch, seq, d_model]
    }
}

#[derive(Config, Debug)]
pub struct FeatureEmbedderConfig {
    cardinalities: Vec<usize>,
    embedding_dims: Vec<usize>,
}

impl FeatureEmbedderConfig {
    pub fn init<B: Backend>(&self) -> FeatureEmbedder<B> {
        let embedders: Vec<Embedding<B>> = self
            .cardinalities
            .iter()
            .zip(self.embedding_dims.iter())
            .map(|(c, e)| EmbeddingConfig::new(*c, *e).init())
            .collect();

        FeatureEmbedder { embedders }
    }
}
