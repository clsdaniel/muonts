
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::InMemDataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Data, Int, Tensor};
use once_cell::sync::OnceCell;
use rand;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use crate::data::batchitem::BatchItem;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GluonTSItem {
    item_id: String,
    target: Vec<f32>,
    observed_values: Option<Vec<f32>>,
    feat_static_cat: Option<Vec<i32>>,
    feat_static_real: Option<Vec<f32>>,
    feat_dynamic_real: Option<Vec<Vec<f32>>>,
    feat_dynamic_cat: Option<Vec<Vec<i32>>>,
}

pub struct DataBatcher<B: Backend> {
    device: B::Device,
    context_length: usize,
    forecast_length: usize,
    seed: u64,
}

impl<B: Backend> DataBatcher<B> {
    pub fn new(
        device: B::Device,
        context_length: usize,
        forecast_length: usize,
        seed: u64,
    ) -> Self {
        Self {
            device,
            context_length,
            forecast_length,
            seed,
        }
    }
}

pub fn load_from_file(filename: &str) -> Result<InMemDataset<GluonTSItem>, std::io::Error> {
    InMemDataset::from_json_rows(filename)
}

static GLOBAL_RNG: OnceCell<Mutex<StdRng>> = OnceCell::new();

impl<B: Backend> Batcher<GluonTSItem, BatchItem<B>> for DataBatcher<B> {
    fn batch(&self, items: Vec<GluonTSItem>) -> BatchItem<B> {
        let batch_size = items.len();
        let total_len = self.context_length + self.forecast_length;

        // TODO: This code is basically similar to Uniform sampling GluonTS does,
        // need to refactor this and add other sampling methods
        let pivots: Vec<usize> = {
            let mut rng = GLOBAL_RNG
                .get_or_init(|| {
                    let rng = StdRng::seed_from_u64(self.seed);
                    Mutex::new(rng)
                })
                .lock()
                .unwrap();

            items
                .iter()
                .map(|item| rng.gen_range(0..(item.target.len() - total_len)))
                .collect()
        };

        let (past_tensors, future_tensors): (Vec<Tensor<B, 2>>, Vec<Tensor<B, 2>>) = items
            .iter()
            .zip(pivots.iter())
            .map(|(item, pivot)| {
                let pivot = *pivot;
                let target_len = item.target.len();

                let data = Data::new(
                    item.target.clone(),
                    burn::tensor::Shape { dims: [target_len] },
                );
                let tensor: Tensor<B, 1> = Tensor::from_data(data.convert());
                let past = tensor.clone().slice([pivot..pivot + self.context_length]);
                let future = tensor.slice([pivot + self.context_length..pivot + total_len]);

                (
                    past.reshape([1, self.context_length]),
                    future.reshape([1, self.forecast_length]),
                )
            })
            .unzip();

        let past_target = Tensor::cat(past_tensors, 0);
        let future_target = Tensor::cat(future_tensors, 0);

        let (past_tensors, future_tensors): (Vec<Tensor<B, 2>>, Vec<Tensor<B, 2>>) = items
            .iter()
            .zip(pivots.iter())
            .map(|(item, pivot)| {
                let pivot = *pivot;
                let target_len = item.target.len();
                match &item.observed_values {
                    Some(observed_values) => {
                        let data = Data::new(
                            observed_values.clone(),
                            burn::tensor::Shape { dims: [target_len] },
                        );
                        let tensor: Tensor<B, 1> = Tensor::from_data(data.convert());
                        let past = tensor.clone().slice([pivot..pivot + self.context_length]);
                        let future = tensor.slice([pivot + self.context_length..pivot + total_len]);

                        (
                            past.reshape([1, self.context_length]),
                            future.reshape([1, self.forecast_length]),
                        )
                    }
                    None => (
                        Tensor::ones([1, self.context_length]),
                        Tensor::ones([1, self.forecast_length]),
                    ),
                }
            })
            .unzip();

        let past_observed_values = Tensor::cat(past_tensors, 0);
        let future_observed_values = Tensor::cat(future_tensors, 0);

        let tensors: Vec<Tensor<B, 2, Int>> = items
            .iter()
            .map(|item| match &item.feat_static_cat {
                Some(values) => {
                    let data = Data::new(
                        values.clone(),
                        burn::tensor::Shape {
                            dims: [values.len()],
                        },
                    );
                    let tensor: Tensor<B, 1, Int> = Tensor::from_data(data.convert());
                    tensor.reshape([1, values.len()])
                }
                None => Tensor::zeros([1, 1]),
            })
            .collect();

        let feat_static_cat = Some(Tensor::cat(tensors, 0));

        // TODO: Currently static reals are ignored
        let feat_static_real: Option<Tensor<B, 2>> = Some(Tensor::zeros([batch_size, 1]));

        let feat_dynamic_real: Vec<Tensor<B, 3>> = 
            items
                .iter()
                .zip(pivots.iter())
                .map(|(item, pivot)| {
                    let pivot = *pivot;
                    let target_len = item.target.len();

                    match &item.feat_dynamic_real {
                        Some(values) => {
                            let feat_count = values.len();

                            let data = Data::new(
                                values.iter().flatten().copied().collect(),
                                burn::tensor::Shape {
                                    dims: [feat_count, target_len],
                                },
                            );
                            let tensor: Tensor<B, 2> = Tensor::from_data(data.convert());

                            // Real features need to be normalized otherwise grandients just explode
                            let tensor = tensor.clone() / (tensor.clone().max_dim(1) + 1e-8);

                            // Got from [feature, seq] to [seq, feature] and finally unsqueeze to get the
                            // final shape of [batch, seq, feature] needed
                            let tensor = tensor.swap_dims(0, 1);
                            let tensor : Tensor<B, 3> = tensor.unsqueeze();

                            tensor.slice([0..1,pivot..pivot+total_len, 0..feat_count])
                        }
                        None => {
                            let zero_tensor: Tensor<B, 3> = Tensor::zeros([
                                1,
                                total_len,
                                1,
                            ]);
                            zero_tensor
                        }
                    }
                })
                .collect();

        let feat_dynamic_real = Tensor::cat(feat_dynamic_real, 0);
        let feat_dynamic_real = Some(feat_dynamic_real);

        // TODO: Currently dynamic categoricals are ignored
        let feat_dynamic_cat = None;
        let past_feat_dynamic_cat = None;
        let past_feat_dynamic_real = None;


        BatchItem {
            past_target,
            past_observed_values,
            future_target,
            future_observed_values,
            feat_static_real,
            feat_static_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
            past_feat_dynamic_cat,
            past_feat_dynamic_real,
        }
    }
}