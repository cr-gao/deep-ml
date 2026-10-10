import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities (per-sample, not normalized).
        pi_theta_ref: List of reference policy probabilities (per-sample, not normalized).
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    # Your code here
    rhos = np.asarray(rhos, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    pi_theta_old = np.asarray(pi_theta_old, dtype=np.float64)
    pi_theta_ref = np.asarray(pi_theta_ref, dtype=np.float64)
    unclipped = rhos * A
    clipped = np.clip(rhos, 1 - epsilon, 1 + epsilon) * A
    obj = np.minimum(unclipped, clipped)
    imp = pi_theta_ref / rhos * pi_theta_old
    kl = rhos * (imp - np.log(imp) - 1)
    obj -= beta * kl
    return obj.mean()