use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, Lstm, LstmConfig};
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
        states: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> Tensor<B, 3> {
        let (cell_state, hidden_state) = self.encoder_lstm.forward(ctx_input.clone(), states);
        let ctx_encodings = hidden_state.clone();

        let last_hidden_state: Tensor<B, 2> = {
            let [batch, d_seq, d_hidden] = hidden_state.dims();
            hidden_state
                .slice([0..batch, d_seq - 1..d_seq, 0..d_hidden])
                .squeeze(1)
        };

        let last_cell_state: Tensor<B, 2> = {
            let [batch, d_seq, d_hidden] = cell_state.dims();
            cell_state
                .slice([0..batch, d_seq - 1..d_seq, 0..d_hidden])
                .squeeze(1)
        };

        let (encodings, skip) = match tgt_input {
            Some(input) => {
                let states = (last_cell_state, last_hidden_state);
                let (_, tgt_encodings) = self.decoder_lstm.forward(input.clone(), Some(states));

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
    pub fn init<B: Backend>(&self) -> TemporalFusionEncoder<B> {
        let skip_proj = if self.d_input != self.d_hidden {
            Some(LinearConfig::new(self.d_input, self.d_hidden).init())
        } else {
            None
        };

        let encoder_lstm = LstmConfig::new(self.d_input, self.d_hidden, true).init();
        let decoder_lstm = LstmConfig::new(self.d_input, self.d_hidden, true).init();
        let gate_linear = LinearConfig::new(self.d_hidden, self.d_hidden * 2).init();
        let gate_glu = GatedLinearUnitConfig::new().init();
        let lnorm = LayerNormConfig::new(self.d_hidden).init();

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
