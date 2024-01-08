use crate::data::batchitem::BatchItem;
use crate::utils::{quantile_loss, weighted_average};
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;
use burn::tensor::{backend::Backend, Data, Shape, Tensor};
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};

use super::feature_embedder::{FeatureEmbedder, FeatureEmbedderConfig};
use super::feature_projector::{FeatureProjector, FeatureProjectorConfig};
use super::grn::{GatedResidualNetwork, GatedResidualNetworkConfig};
use super::tfd::{TemportalFusionDecoder, TemportalFusionDecoderConfig};
use super::tfe::{TemporalFusionEncoder, TemporalFusionEncoderConfig};
use super::vsn::{VariableSelectionNetwork, VariableSelectionNetworkConfig};

#[derive(Module, Debug)]
pub struct TemporalFusionTransformerModel<B: Backend> {
    context_length: usize,
    prediction_length: usize,
    quantiles: Tensor<B, 1>,
    target_proj: Linear<B>,
    past_feat_dynamic_proj: Option<FeatureProjector<B>>,
    past_feat_dynamic_embed: Option<FeatureEmbedder<B>>,
    feat_dynamic_proj: Option<FeatureProjector<B>>,
    feat_dynamic_embed: Option<FeatureEmbedder<B>>,
    feat_static_proj: Option<FeatureProjector<B>>,
    feat_static_embed: Option<FeatureEmbedder<B>>,
    static_selector: VariableSelectionNetwork<B>,
    ctx_selector: VariableSelectionNetwork<B>,
    tgt_selector: VariableSelectionNetwork<B>,
    selection: GatedResidualNetwork<B>,
    enrichment: GatedResidualNetwork<B>,
    state_h: GatedResidualNetwork<B>,
    state_c: GatedResidualNetwork<B>,
    temporal_encoder: TemporalFusionEncoder<B>,
    temporal_decoder: TemportalFusionDecoder<B>,
    output_proj: Linear<B>,
}

impl<B: Backend> TemporalFusionTransformerModel<B> {
    fn scaler(
        &self,
        data: Tensor<B, 2>,
        weights: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let data = data.detach().set_require_grad(false);
        let weights = weights.detach().set_require_grad(false);

        let denominator = weights.clone().sum_dim(1);
        let denominator = denominator.clamp_min(1.0);
        let loc = (data.clone() * weights.clone()).sum_dim(1) / denominator.clone();
        let variance = (data.clone() - loc.clone()) * weights;
        let variance = (variance.clone() * variance.clone()).sum_dim(1) / denominator;

        let scale = (variance + 1e-5).sqrt();
        let scaled_data = (data - loc.clone()) / scale.clone();

        (scaled_data, loc, scale)
    }

    pub fn forward(
        &self,
        past_target: Tensor<B, 2>,                        // [N, T]
        past_observed_values: Tensor<B, 2>,               // [N, T]
        feat_static_real: Option<Tensor<B, 2>>,           // [N, D_sr]
        feat_static_cat: Option<Tensor<B, 2, Int>>,       // [N, D_sc]
        feat_dynamic_real: Option<Tensor<B, 3>>,          // [N, T + H, D_dr]
        feat_dynamic_cat: Option<Tensor<B, 3, Int>>,      // = None,  # [N, T + H, D_dc]
        past_feat_dynamic_real: Option<Tensor<B, 3>>,     // = None,  # [N, T, D_pr]
        past_feat_dynamic_cat: Option<Tensor<B, 3, Int>>, //= None,  # [N, T, D_pc]
    ) -> Tensor<B, 3> {
        let (past_target, loc, scale) = self.scaler(past_target, past_observed_values.clone());
        let loc: Tensor<B, 3> = loc.unsqueeze_dim(1);
        let scale: Tensor<B, 3> = scale.unsqueeze_dim(1);
        let mut past_covariates: Vec<Tensor<B, 3>> =
            vec![self.target_proj.forward(past_target.unsqueeze_dim(2))];
        let mut future_covariates: Vec<Tensor<B, 3>> = Vec::new();
        let mut static_covariates: Vec<Tensor<B, 2>> = Vec::new();

        if let Some(past_feat_dynamic_proj) = &self.past_feat_dynamic_proj {
            let mut projs = past_feat_dynamic_proj.forward(past_feat_dynamic_real.unwrap());
            past_covariates.append(&mut projs);
        }

        if let Some(past_feat_dynamic_embed) = &self.past_feat_dynamic_embed {
            let mut embs = past_feat_dynamic_embed.forward(past_feat_dynamic_cat.unwrap());
            past_covariates.append(&mut embs);
        }

        if let Some(feat_dynamic_proj) = &self.feat_dynamic_proj {
            let projs = feat_dynamic_proj.forward(feat_dynamic_real.unwrap());
            for proj in projs {
                let [dim_a, dim_b, dim_c] = proj.dims();
                let ctx_proj = proj
                    .clone()
                    .slice([0..dim_a, 0..self.context_length, 0..dim_c]);
                let tgt_proj = proj.slice([0..dim_a, self.context_length..dim_b, 0..dim_c]);

                past_covariates.push(ctx_proj);
                future_covariates.push(tgt_proj);
            }
        }

        if let Some(feat_dynamic_embed) = &self.feat_dynamic_embed {
            let embs = feat_dynamic_embed.forward(feat_dynamic_cat.unwrap());
            for emb in embs {
                let [dim_a, dim_b, dim_c] = emb.dims();
                let ctx_emb = emb
                    .clone()
                    .slice([0..dim_a, 0..self.context_length, 0..dim_c]);
                let tgt_emb = emb.slice([0..dim_a, self.context_length..dim_b, 0..dim_c]);

                past_covariates.push(ctx_emb);
                future_covariates.push(tgt_emb);
            }
        }

        if let Some(feat_static_proj) = &self.feat_static_proj {
            let mut projs = feat_static_proj.forward(feat_static_real.unwrap());
            static_covariates.append(&mut projs);
        }

        if let Some(feat_static_embed) = &self.feat_static_embed {
            let mut embs: Vec<Tensor<B, 2>> = feat_static_embed
                .forward(feat_static_cat.unwrap())
                .into_iter()
                .map(|x| x.squeeze(1))
                .collect();

            static_covariates.append(&mut embs);
        }

        let (static_var, _) = self.static_selector.forward(static_covariates, None);

        let c_selection: Tensor<B, 3> = self
            .selection
            .forward(static_var.clone(), None)
            .unsqueeze_dim(1);
        let c_enrichment = self
            .enrichment
            .forward(static_var.clone(), None)
            .unsqueeze_dim(1);

        let c_h = self.state_h.forward(static_var.clone(), None);
        let c_c = self.state_c.forward(static_var, None);

        let states = (c_c, c_h);

        let (ctx_input, _) = self
            .ctx_selector
            .forward(past_covariates, Some(c_selection.clone()));
        let (tgt_input, _) = self
            .tgt_selector
            .forward(future_covariates, Some(c_selection));

        let encoding = self
            .temporal_encoder
            .forward(ctx_input, Some(tgt_input), Some(states));
        let decoding = self
            .temporal_decoder
            .forward(encoding, c_enrichment, past_observed_values);
        let preds = self.output_proj.forward(decoding);
        let output = preds * scale + loc;

        output.swap_dims(1, 2)
    }

    pub fn forward_regression(
        &self,
        past_target: Tensor<B, 2>,                        // [N, T]
        past_observed_values: Tensor<B, 2>,               // [N, T]
        future_target: Tensor<B, 2>,                      // [N, T]
        future_observed_values: Tensor<B, 2>,             // [N, T]
        feat_static_real: Option<Tensor<B, 2>>,           // [N, D_sr]
        feat_static_cat: Option<Tensor<B, 2, Int>>,       // [N, D_sc]
        feat_dynamic_real: Option<Tensor<B, 3>>,          // [N, T + H, D_dr]
        feat_dynamic_cat: Option<Tensor<B, 3, Int>>,      // = None,  # [N, T + H, D_dc]
        past_feat_dynamic_real: Option<Tensor<B, 3>>,     // = None,  # [N, T, D_pr]
        past_feat_dynamic_cat: Option<Tensor<B, 3, Int>>, //= None,  # [N, T, D_pc]
    ) -> RegressionOutput<B> {
        let output_targets = future_target.clone();

        let pred = self.forward(
            past_target,
            past_observed_values,
            feat_static_real,
            feat_static_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
        );
        let output_pred = pred.clone().mean_dim(2).squeeze(2);
        let loss = quantile_loss(future_target, pred.swap_dims(1, 2), self.quantiles.clone());
        let loss = weighted_average(loss, future_observed_values);

        RegressionOutput::new(loss, output_pred, output_targets)
    }
}

impl<B: AutodiffBackend> TrainStep<BatchItem<B>, RegressionOutput<B>>
    for TemporalFusionTransformerModel<B>
{
    fn step(&self, batch: BatchItem<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(
            batch.past_target,
            batch.past_observed_values,
            batch.future_target,
            batch.future_observed_values,
            batch.feat_static_real,
            batch.feat_static_cat,
            batch.feat_dynamic_real,
            batch.feat_dynamic_cat,
            batch.past_feat_dynamic_real,
            batch.past_feat_dynamic_cat,
        );
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<BatchItem<B>, RegressionOutput<B>>
    for TemporalFusionTransformerModel<B>
{
    fn step(&self, batch: BatchItem<B>) -> RegressionOutput<B> {
        self.forward_regression(
            batch.past_target,
            batch.past_observed_values,
            batch.future_target,
            batch.future_observed_values,
            batch.feat_static_real,
            batch.feat_static_cat,
            batch.feat_dynamic_real,
            batch.feat_dynamic_cat,
            batch.past_feat_dynamic_real,
            batch.past_feat_dynamic_cat,
        )
    }
}

#[derive(Config, Debug)]
pub struct TemporalFusionTransformerModelConfig {
    context_length: usize,
    prediction_length: usize,

    #[config(default = "vec![0.1, 0.5, 0.9]")]
    quantiles: Vec<f32>,

    #[config(default = 4)]
    num_heads: usize,

    #[config(default = 32)]
    d_hidden: usize,

    #[config(default = 32)]
    d_var: usize,

    #[config(default = 0.1)]
    dropout: f64,

    #[config(default = "vec![1]")]
    d_feat_static_real: Vec<usize>,

    #[config(defualt = "vec![1]")]
    c_feat_static_cat: Vec<usize>,

    #[config(defualt = "vec![1]")]
    d_feat_dynamic_real: Vec<usize>,

    #[config(defualt = "vec![]")]
    c_feat_dynamic_cat: Vec<usize>,

    #[config(defualt = "vec![]")]
    d_past_feat_dynamic_real: Vec<usize>,

    #[config(defualt = "vec![]")]
    c_past_feat_dynamic_cat: Vec<usize>,
}

impl TemporalFusionTransformerModelConfig {
    pub fn init<B: Backend>(&self) -> TemporalFusionTransformerModel<B> {
        let num_feat_static = self.d_feat_static_real.len() + self.c_feat_static_cat.len();
        let num_feat_dynamic = self.d_feat_dynamic_real.len() + self.c_feat_dynamic_cat.len();
        let num_past_feat_dynamic =
            self.d_past_feat_dynamic_real.len() + self.c_past_feat_dynamic_cat.len();

        let target_proj = LinearConfig::new(1, self.d_var).init();

        let past_feat_dynamic_proj = if self.d_past_feat_dynamic_real.len() > 0 {
            Some(
                FeatureProjectorConfig::new(
                    self.d_past_feat_dynamic_real.clone(),
                    self.d_past_feat_dynamic_real
                        .iter()
                        .map(|_| self.d_var)
                        .collect(),
                )
                .init(),
            )
        } else {
            None
        };

        let past_feat_dynamic_embed = if self.c_past_feat_dynamic_cat.len() > 0 {
            Some(
                FeatureEmbedderConfig::new(
                    self.c_past_feat_dynamic_cat.clone(),
                    self.c_past_feat_dynamic_cat
                        .iter()
                        .map(|_| self.d_var)
                        .collect(),
                )
                .init(),
            )
        } else {
            None
        };

        let feat_dynamic_proj = if self.d_feat_dynamic_real.len() > 0 {
            Some(
                FeatureProjectorConfig::new(
                    self.d_feat_dynamic_real.clone(),
                    self.d_feat_dynamic_real
                        .iter()
                        .map(|_| self.d_var)
                        .collect(),
                )
                .init(),
            )
        } else {
            None
        };

        let feat_dynamic_embed = if self.c_feat_dynamic_cat.len() > 0 {
            Some(
                FeatureEmbedderConfig::new(
                    self.c_feat_dynamic_cat.clone(),
                    self.c_feat_dynamic_cat.iter().map(|_| self.d_var).collect(),
                )
                .init(),
            )
        } else {
            None
        };

        let feat_static_proj = if self.d_feat_static_real.len() > 0 {
            Some(
                FeatureProjectorConfig::new(
                    self.d_feat_static_real.clone(),
                    self.d_feat_static_real.iter().map(|_| self.d_var).collect(),
                )
                .init(),
            )
        } else {
            None
        };

        let feat_static_embed = if self.c_feat_static_cat.len() > 0 {
            Some(
                FeatureEmbedderConfig::new(
                    self.c_feat_static_cat.clone(),
                    self.c_feat_static_cat.iter().map(|_| self.d_var).collect(),
                )
                .init(),
            )
        } else {
            None
        };

        let static_selector = VariableSelectionNetworkConfig::new(self.d_var, num_feat_static)
            .with_dropout(self.dropout)
            .init();

        let ctx_selector = VariableSelectionNetworkConfig::new(
            self.d_var,
            num_past_feat_dynamic + num_feat_dynamic + 1,
        )
        .with_add_static(true)
        .with_dropout(self.dropout)
        .init();

        let tgt_selector = VariableSelectionNetworkConfig::new(self.d_var, num_feat_dynamic)
            .with_add_static(true)
            .with_dropout(self.dropout)
            .init();

        let selection = GatedResidualNetworkConfig::new(self.d_var)
            .with_dropout(self.dropout)
            .init();

        let enrichment = GatedResidualNetworkConfig::new(self.d_var)
            .with_dropout(self.dropout)
            .init();

        let state_h = GatedResidualNetworkConfig::new(self.d_var)
            .with_d_output(Some(self.d_hidden))
            .with_dropout(self.dropout)
            .init();

        let state_c = GatedResidualNetworkConfig::new(self.d_var)
            .with_d_output(Some(self.d_hidden))
            .with_dropout(self.dropout)
            .init();

        let temporal_encoder = TemporalFusionEncoderConfig::new(self.d_var, self.d_hidden).init();
        let temporal_decoder = TemportalFusionDecoderConfig::new(
            self.context_length,
            self.prediction_length,
            self.d_hidden,
            self.d_var,
            self.num_heads,
        )
        .with_dropout(self.dropout)
        .init();

        let output_proj = LinearConfig::new(self.d_hidden, self.quantiles.len()).init();

        let quantiles = Data::new(self.quantiles.clone(), Shape::new([self.quantiles.len()]));
        let quantiles: Tensor<B, 1> = Tensor::from_data(quantiles.convert());

        TemporalFusionTransformerModel {
            context_length: self.context_length,
            prediction_length: self.prediction_length,
            quantiles,
            target_proj,
            past_feat_dynamic_proj,
            past_feat_dynamic_embed,
            feat_dynamic_proj,
            feat_dynamic_embed,
            feat_static_proj,
            feat_static_embed,
            static_selector,
            ctx_selector,
            tgt_selector,
            selection,
            state_h,
            enrichment,
            state_c,
            temporal_encoder,
            temporal_decoder,
            output_proj,
        }
    }
}
