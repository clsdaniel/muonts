use burn::tensor::{backend::Backend, Int, Tensor};

#[derive(Clone, Debug)]
pub struct BatchItem<B: Backend> {
    pub past_target: Tensor<B, 2>,                        // [N, T]
    pub past_observed_values: Tensor<B, 2>,               // [N, T]
    pub future_target: Tensor<B, 2>,                      // [N, T]
    pub future_observed_values: Tensor<B, 2>,             // [N, T]
    pub feat_static_real: Option<Tensor<B, 2>>,           // [N, D_sr]
    pub feat_static_cat: Option<Tensor<B, 2, Int>>,       // [N, D_sc]
    pub feat_dynamic_real: Option<Tensor<B, 3>>,          // [N, T + H, D_dr]
    pub feat_dynamic_cat: Option<Tensor<B, 3, Int>>,      // = None,  # [N, T + H, D_dc]
    pub past_feat_dynamic_real: Option<Tensor<B, 3>>,     // = None,  # [N, T, D_pr]
    pub past_feat_dynamic_cat: Option<Tensor<B, 3, Int>>, //= None,  # [N, T, D_pc]
}
