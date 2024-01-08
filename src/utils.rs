use burn::tensor::{backend::Backend, Tensor};

pub fn weighted_average<B: Backend>(x: Tensor<B, 2>, weights: Tensor<B, 2>) -> Tensor<B, 1> {
    let zeros = x.zeros_like();
    let mask = weights.clone().equal_elem(0.0).bool_not();
    let weighted = x * weights.clone();

    let weighted_tensor = zeros.mask_where(mask, weighted);
    let sum_weights = weights.sum().clamp_min(1.0);

    weighted_tensor.sum() / sum_weights
}

pub fn quantile_loss<B: Backend>(
    y_true: Tensor<B, 2>,
    y_pred: Tensor<B, 3>,
    quantiles: Tensor<B, 1>,
) -> Tensor<B, 2> {
    let quantiles: Tensor<B, 3> = quantiles.unsqueeze();
    let y_true: Tensor<B, 3> = y_true.unsqueeze_dim(2);
    let residual = y_true.clone() - y_pred.clone();
    let loss = y_pred.lower_equal(y_true).float() - quantiles;
    (residual * loss).abs().sum_dim(2).squeeze(2)
}

pub fn split<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    splits: Vec<usize>,
    dim: i32,
) -> Vec<Tensor<B, D>> {
    let dim: usize = if dim < 0 { D - 1 } else { dim as usize };

    let dim_size = x.dims()[dim];

    if splits.len() == 0 {
        return vec![x];
    }

    if splits.len() > 1 {
        assert!(splits.iter().copied().fold(0, |x, y| x + y) == dim_size);
    }

    let splits: Vec<usize> = if splits.len() == 1 {
        let l = splits[0];
        let reps = dim_size / l;
        let reminder = dim_size % l;
        (0..reps).map(|_| l).chain(vec![reminder]).collect()
    } else {
        splits
    };

    let mut current_idx: usize = 0;
    splits
        .into_iter()
        .map(|s| {
            let mut ranges = x.dims().map(|x| 0..x);
            ranges[dim] = current_idx..current_idx + s;
            current_idx += s;
            x.clone().slice(ranges)
        })
        .collect()
}
