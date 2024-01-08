use burn::config::Config;
use burn::module::Module;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{backend::Backend, Tensor};

use super::glu::{GatedLinearUnit, GatedLinearUnitConfig};
use super::grn::{GatedResidualNetwork, GatedResidualNetworkConfig};

#[derive(Module, Debug)]
pub struct TemportalFusionDecoder<B: Backend> {
    context_length: usize,
    prediction_length: usize,
    enrich: GatedResidualNetwork<B>,
    attention: MultiHeadAttention<B>,
    att_net_linear: Linear<B>,
    att_net_glu: GatedLinearUnit,
    att_lnorm: LayerNorm<B>,
    ff_net_grn: GatedResidualNetwork<B>,
    ff_net_linear: Linear<B>,
    ff_net_glu: GatedLinearUnit,
    ff_lnorm: LayerNorm<B>,
}

impl<B: Backend> TemportalFusionDecoder<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        statics: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let expanded_static = statics.repeat(1, self.context_length + self.prediction_length);
        let skip = {
            let [batch_size, seq_size, val_size] = x.dims();
            x.clone()
                .slice([0..batch_size, self.context_length..seq_size, 0..val_size])
        };
        let x = self.enrich.forward(x, Some(expanded_static));

        let mask_pad = {
            let mask_pad = mask.ones_like();
            let [x, _] = mask_pad.dims();
            let mask_pad = mask_pad.slice([0..x, 0..1]);
            mask_pad.repeat(1, self.prediction_length)
        };

        let key_padding_mask = (Tensor::cat(vec![mask, mask_pad], 1).neg() + 1.0)
            .equal_elem(0)
            .bool_not();

        let query = {
            let [a, b, c] = x.dims();
            x.clone().slice([0..a, self.context_length..b, 0..c])
        };

        let att_input = MhaInput::new(query, x.clone(), x.clone()).mask_pad(key_padding_mask);
        let mha_output = self.attention.forward(att_input);
        let attn_output = mha_output.context;

        let att = self.att_net_linear.forward(attn_output);
        let att = self.att_net_glu.forward(att);

        let x = {
            let [a, b, c] = x.dims();
            x.slice([0..a, self.context_length..b, 0..c])
        };
        let x = self.att_lnorm.forward(x + att);
        let x = self.ff_net_grn.forward(x, None);
        let x = self.ff_net_linear.forward(x);
        let x = self.ff_net_glu.forward(x);
        let x = self.ff_lnorm.forward(x + skip);

        x
    }
}

#[derive(Config, Debug)]
pub struct TemportalFusionDecoderConfig {
    context_length: usize,
    prediction_length: usize,
    d_hidden: usize,
    d_var: usize,
    num_heads: usize,

    #[config(default = 0.0)]
    dropout: f64,
}

impl TemportalFusionDecoderConfig {
    pub fn init<B: Backend>(&self) -> TemportalFusionDecoder<B> {
        let enrich = GatedResidualNetworkConfig::new(self.d_hidden)
            .with_d_static(self.d_var)
            .with_dropout(self.dropout)
            .init();

        let attention = MultiHeadAttentionConfig::new(self.d_hidden, self.num_heads)
            .with_dropout(self.dropout)
            .init();

        let att_net_linear = LinearConfig::new(self.d_hidden, self.d_hidden * 2).init();
        let att_net_glu = GatedLinearUnitConfig::new().init();
        let att_lnorm = LayerNormConfig::new(self.d_hidden).init();

        let ff_net_grn = GatedResidualNetworkConfig::new(self.d_hidden)
            .with_dropout(self.dropout)
            .init();
        let ff_net_linear = LinearConfig::new(self.d_hidden, self.d_hidden * 2).init();
        let ff_net_glu = GatedLinearUnitConfig::new().init();
        let ff_lnorm = LayerNormConfig::new(self.d_hidden).init();

        TemportalFusionDecoder {
            context_length: self.context_length,
            prediction_length: self.prediction_length,
            enrich,
            attention,
            att_net_linear,
            att_net_glu,
            att_lnorm,
            ff_net_grn,
            ff_net_linear,
            ff_net_glu,
            ff_lnorm,
        }
    }
}
