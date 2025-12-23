import numpy as np


def stratified_time_sampling(
    indices,
    total_len,
    n_samples,
    n_bins=10
):
    bin_size = total_len // n_bins
    sampled = []

    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, total_len)

        candidates = [
            idx for idx in indices
            if start <= idx < end
        ]

        if candidates:
            sampled.extend(
                np.random.choice(
                    candidates,
                    min(len(candidates), max(1, n_samples // n_bins)),
                    replace=False
                ).tolist()
            )

    return sampled
