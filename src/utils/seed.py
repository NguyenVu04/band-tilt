"""Global seed control.

Seeding happens in exactly one place so that "which seed produced this?" always
has an answer: ``cfg.seed`` in ``configs/config.yaml``. Call :func:`set_seed`
at the top of every notebook and every entry point, before anything random
happens.

A literal seed passed to a splitter, a sampler or an estimator anywhere else in
the project is a bug — it silently decouples that component from the config.
"""


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every random source in use.

    Args:
        seed: The value from ``cfg.seed``.
        deterministic: Also force deterministic GPU kernels. Reproducible but
            slower, and some operations have no deterministic implementation.

    Raises:
        NotImplementedError: Always — implement this module first.

    Notes:
        Cover ``random``, ``numpy``, ``PYTHONHASHSEED``, and the DL framework if
        one is installed — import it inside the function so the base
        environment does not need it.

        Seeding is necessary but not sufficient for reproducibility: thread
        counts, library versions and GPU nondeterminism all still matter, which
        is why the config and the lock file are versioned alongside the code.

    Example:
        >>> set_seed(cfg.seed)
    """
    # TODO(1): random.seed(seed); np.random.seed(seed)
    # TODO(2): os.environ["PYTHONHASHSEED"] = str(seed)
    # TODO(3): torch.manual_seed / cuda.manual_seed_all inside a try/ImportError
    # TODO(4): if deterministic: torch.use_deterministic_algorithms(True)
    raise NotImplementedError("src.utils.seed.set_seed")
