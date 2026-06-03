"""Pick a deterministic sklearn PCA ``svd_solver``.

sklearn's default ``svd_solver='auto'`` often selects the stochastic ``'randomized'``
solver for both ``compute_pca_features`` during phase-1 clustering and
``SnippetClassifier.fit`` during per-channel classifier construction, which makes
sorting non-reproducible run to run on identical inputs if not seeded.

We want:
- ``'covariance_eigh'`` when ``n_features`` is modest and ``n_samples >= n_features``,
  since it is exact, deterministic, and fastest in that regime. This regime virtually
  always applies when fitting per-channel classifiers, regardless of whether we are
  sorting tetrode data or Neuropixel data.
- ``'full'`` when ``n_samples < n_features`` (and ``n_features`` is modest), since it is
  exact, deterministic, numerically stable, and cheap (bounded by
  ``min(n_samples, n_features)``). That said, this regime almost never applies.
- ``'randomized'`` when ``n_features`` exceeds ``cov_cap``, since materializing the
  ``n_features x n_features`` covariance matrix and its eigendecomposition is
  impractical -- e.g. a whole-probe Neuropixels phase-1 PCA can have
  ``n_features ~= 21k``. That's ~3.6 GB covariance and even more for the
  eigendecomposition workspace, plus many minutes (tens) of compute. ``'randomized'`` is
  what ``'auto'`` already picks in that case; Yes, it stays approximate, but seeding it
  at least makes it deterministic.

The ``cov_cap`` default of 8000 corresponds to a ~0.5 GB covariance matrix. It is a
safety rail: realistic 64ch tetrode (``n_features ~= 160``) and Neuropixels
(``n_features <~ 1600``) classifiers never approach it, and only the whole-probe
``compute_pca_features`` call exceeds it.

NOTE: callers MUST pass ``random_state=<int>`` to ``PCA`` so the ``'randomized'``
branch is deterministic. ``random_state`` is ignored by the exact solvers, so it is
safe to pass unconditionally.
"""


def deterministic_pca_solver(
    n_samples: int, n_features: int, *, cov_cap: int = 8000
) -> str:
    """Return an sklearn PCA ``svd_solver`` that is never the unseeded ``'randomized'`` auto-pick.

    See module docstring for the rationale. Callers must still pass ``random_state``
    to ``PCA`` so the ``'randomized'`` branch (large ``n_features``) is reproducible.
    """
    if n_features > cov_cap:
        return "randomized"
    if n_samples >= n_features:
        return "covariance_eigh"
    return "full"
