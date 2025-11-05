use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, Lstm, LstmConfig, LstmState};
use burn::tensor::{backend::Backend, Tensor};
use super::glu::{GatedLinearUnit, GatedLinearUnitConfig};

#[derive(Module, Debug)]
pub struct TemporalFusionEncoder<B: Backend> {
    encoder_lstm: Lstm<B>,
    decoder_lstm: Lstm<B>,
    gate_linear: Linear<B>,
    gate_glu: GatedLinearUnit,
    skip_proj: Option<Linear<B>>,
    lnorm: LayerNorm<B>,
}

impl<B: Backend> TemporalFusionEncoder<B> {
    pub fn forward(
        &self,
        ctx_input: Tensor<B, 3>,
        tgt_input: Option<Tensor<B, 3>>,
        states: Option<LstmState<B, 2>>,
    ) -> Tensor<B, 3> {
        let (hidden_state, final_state) = self.encoder_lstm.forward(ctx_input.clone(), states);
        let ctx_encodings = hidden_state.clone();

        let (encodings, skip) = match tgt_input {
            Some(input) => {
                let states = Some(final_state);
                let (tgt_encodings, _) = self.decoder_lstm.forward(input.clone(), states);

                let encodings = Tensor::cat(vec![ctx_encodings, tgt_encodings], 1);
                let skip = Tensor::cat(vec![ctx_input, input], 1);

                (encodings, skip)
            }
            None => (ctx_encodings, ctx_input),
        };

        let encodings = self.gate_linear.forward(encodings);
        let encodings = self.gate_glu.forward(encodings);

        let skip = match &self.skip_proj {
            Some(proj) => proj.forward(skip),
            None => skip,
        };

        let encodings = self.lnorm.forward(encodings + skip);

        encodings
    }
}

#[derive(Config, Debug)]
pub struct TemporalFusionEncoderConfig {
    d_input: usize,
    d_hidden: usize,
}

impl TemporalFusionEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TemporalFusionEncoder<B> {
        let skip_proj = if self.d_input != self.d_hidden {
            Some(LinearConfig::new(self.d_input, self.d_hidden).init(device))
        } else {
            None
        };

        let encoder_lstm = LstmConfig::new(self.d_input, self.d_hidden, true).init(device);
        let decoder_lstm = LstmConfig::new(self.d_input, self.d_hidden, true).init(device);
        let gate_linear = LinearConfig::new(self.d_hidden, self.d_hidden * 2).init(device);
        let gate_glu = GatedLinearUnitConfig::new().init();
        let lnorm = LayerNormConfig::new(self.d_hidden).init(device);

        TemporalFusionEncoder {
            encoder_lstm,
            decoder_lstm,
            gate_linear,
            gate_glu,
            skip_proj,
            lnorm,
        }
    }
}
