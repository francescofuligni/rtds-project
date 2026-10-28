"""Sample private training rows and disjoint public/holdout rows."""

from rashomon.data import BinarizedDataset


def sample_private_public_indices(
    data: BinarizedDataset, *, n: int, public_n: int, seed: int
):
    """Preserve the seeded private sample and disjoint public/holdout sample."""
    private = data.X.sample(n=n, random_state=seed).index
    public = (
        data.X.drop(index=private).sample(n=public_n, random_state=seed + 10_000).index
    )
    return private, public
