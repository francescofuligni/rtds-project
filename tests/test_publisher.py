import pandas as pd

from rashomon.data import BinarizedDataset
from rashomon.publisher import sample_private_public_indices


def test_sampling_preserves_original_indices_and_disjointness():
    data = BinarizedDataset(
        pd.DataFrame({"x": range(20)}), pd.Series(range(20)), {}, {}
    )
    private, public = sample_private_public_indices(data, n=5, public_n=7, seed=3)
    expected = data.X.sample(n=5, random_state=3).index
    assert private.equals(expected)
    assert public.equals(
        data.X.drop(index=expected).sample(n=7, random_state=10003).index
    )
    assert set(private).isdisjoint(public)
