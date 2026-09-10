"""Fixed-thread runtime configuration for analytical runs.

The repository-wide production mode defaults to one thread.  The committed
JAMA-scoped entry point sets ``MEDICAL_FAIRNESS_N_JOBS=8`` before importing
numerical libraries, giving that workflow a fixed, recorded eight-thread
profile without permitting unrestricted parallelism.
"""

import os


def _configured_n_jobs():
    raw = os.environ.get("MEDICAL_FAIRNESS_N_JOBS", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("MEDICAL_FAIRNESS_N_JOBS must be a positive integer") from exc
    if value < 1:
        raise RuntimeError("MEDICAL_FAIRNESS_N_JOBS must be a positive integer")
    return value


DETERMINISTIC_N_JOBS = _configured_n_jobs()
THREAD_ENVIRONMENT = {
    "BLAS_NUM_THREADS": str(DETERMINISTIC_N_JOBS),
    "OMP_NUM_THREADS": str(DETERMINISTIC_N_JOBS),
    "MKL_NUM_THREADS": str(DETERMINISTIC_N_JOBS),
    "OPENBLAS_NUM_THREADS": str(DETERMINISTIC_N_JOBS),
    "NUMEXPR_NUM_THREADS": str(DETERMINISTIC_N_JOBS),
    "BLIS_NUM_THREADS": str(DETERMINISTIC_N_JOBS),
    "VECLIB_MAXIMUM_THREADS": str(DETERMINISTIC_N_JOBS),
}


def configure_deterministic_environment():
    """Force native numerical thread limits before numerical imports."""
    for name, value in THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    return dict(THREAD_ENVIRONMENT)


def enforce_loaded_threadpool_limits():
    """Also cap numerical libraries that were loaded before this module."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=DETERMINISTIC_N_JOBS)


def runtime_provenance():
    """Return the effective deterministic settings for a run manifest."""
    from threadpoolctl import threadpool_info

    return {
        "n_jobs": DETERMINISTIC_N_JOBS,
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "loaded_threadpools": [
            {
                "internal_api": item.get("internal_api"),
                "user_api": item.get("user_api"),
                "prefix": item.get("prefix"),
                "num_threads": item.get("num_threads"),
                "version": item.get("version"),
            }
            for item in threadpool_info()
        ],
    }
