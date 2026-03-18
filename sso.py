import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


__all__ = ["SSO"]

# Newton-Schulz utilities (Polar-Express coefficients)

# 8-step Polar-Express coefficients for matrix sign via Newton-Schulz iteration.
# Matches Megatron-LM spectral_ball_utils._small_msign.
_POLAR_EXPRESS_COEFFS: List[Tuple[float, float, float]] = [
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612,  0.5184),
    (3.9096, -2.8234,  0.5250),
    (3.2856, -2.4153,  0.4853),
    (2.2779, -1.6198,  0.3985),
    (1.8726, -1.2307,  0.3585),
    (1.8564, -1.2132,  0.3568),
    (1.8750, -1.2500,  0.3750),
]


@torch.no_grad()
def _ns_step(X: Tensor, a: float, b: float, c: float) -> Tensor:
    """One Newton-Schulz step: X <- a*X + (b*A + c*A^2) @ X, where A = X @ X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    return torch.addmm(X, B, X, alpha=1.0, beta=a)


@torch.no_grad()
def _msign(G: Tensor, steps: int = 8, eps: float = 1e-7) -> Tensor:
    """Matrix sign function via Newton-Schulz with Polar-Express coefficients.

    Computes the orthogonal factor U @ V^T of the SVD of G (i.e. the matrix sign
    of G in the sense of polar decomposition). Uses bfloat16 for GPU efficiency.

    Args:
        G: Input matrix (2-D tensor).
        steps: Number of Newton-Schulz iterations (default: 8).
        eps: Numerical stability epsilon.

    Returns:
        Orthogonalized matrix of the same shape as G, cast back to G's dtype.
    """
    X = G.bfloat16()
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / X.norm().clamp_min(eps)
    for i in range(steps):
        a, b, c = _POLAR_EXPRESS_COEFFS[min(i, len(_POLAR_EXPRESS_COEFFS) - 1)]
        X = _ns_step(X, a, b, c)
    if transpose:
        X = X.T
    return X.to(G.dtype)


@torch.no_grad()
def _power_iteration(
    W: Tensor, steps: int = 5, eps: float = 1e-20
) -> Tuple[Tensor, Tensor, Tensor]:
    """Bilateral power iteration for the leading singular triplet (sigma, u, v).

    Args:
        W: 2-D weight matrix (float32).
        steps: Number of power iterations.
        eps: Epsilon for normalization.

    Returns:
        (sigma, u, v): Leading singular value and left/right singular vectors.
    """
    W = W.to(torch.float32)
    v = torch.ones(W.shape[1], 1, device=W.device, dtype=torch.float32)
    for _ in range(steps):
        u = torch.nn.functional.normalize(W @ v, dim=0, eps=eps)
        v = torch.nn.functional.normalize(W.T @ u, dim=0, eps=eps)
    sigma = (u.T @ W @ v).squeeze()
    return sigma, u, v


# ---------------------------------------------------------------------------
# Retraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _apply_retract(W: Tensor, sigma: float, target_radius: float, eps: float = 1e-8) -> None:
    """Hard spectral-norm retraction: W <- (R / sigma) * W  (in-place).

    Projects W back onto the spectral ball of radius ``target_radius``.
    """
    scale = target_radius / (max(sigma, 0.0) + eps)
    if abs(scale - 1.0) > eps:
        W.mul_(scale)


# ---------------------------------------------------------------------------
# Lambda solver (tangent-space constraint)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _f_lambda(G: Tensor, Theta: Tensor, lam: float, msign_steps: int) -> float:
    """Evaluate f(lam) = <Theta, msign(G + lam * Theta)>."""
    return float((Theta * _msign(G + lam * Theta, steps=msign_steps)).sum())


@torch.no_grad()
def _solve_lambda(
    G: Tensor,
    Theta: Tensor,
    init: float = 0.0,
    init_step: float = 1e-3,
    tol: float = 1e-6,
    max_iter: int = 20,
    max_expand: int = 10,
    msign_steps: int = 8,
) -> float:
    """Solve lam* s.t. f(lam*) = <Theta, msign(G + lam* Theta)> = 0.

    Uses adaptive bracket search followed by bisection.

    Args:
        G: Normalized momentum matrix.
        Theta: Normal direction (u @ v^T).
        init: Initial guess for lambda.
        init_step: Initial bracket step size.
        tol: Convergence tolerance on |f(lam)|.
        max_iter: Maximum bisection iterations.
        max_expand: Maximum bracket expansion steps.
        msign_steps: Newton-Schulz steps inside f evaluation.

    Returns:
        lam*: Scalar float satisfying the tangent constraint.
    """
    f0 = _f_lambda(G, Theta, init, msign_steps)
    if abs(f0) < tol:
        return init

    step = init_step if f0 < 0 else -init_step
    lam_a, f_a = init, f0
    lam_b, f_b = init, f0

    # Bracket search
    for _ in range(max_expand):
        lam_new = lam_a + step
        f_new = _f_lambda(G, Theta, lam_new, msign_steps)
        if (f_a <= 0) != (f_new <= 0):
            lam_b, f_b = lam_new, f_new
            if f_a > 0:
                lam_a, f_a, lam_b, f_b = lam_b, f_b, lam_a, f_a
            break
        step *= 2.0
        lam_a, f_a = lam_new, f_new
    else:
        return init  # bracket failed, fall back

    best_lam = lam_a if abs(f_a) <= abs(f_b) else lam_b
    best_f = min(abs(f_a), abs(f_b))

    # Bisection
    for _ in range(max_iter):
        lam_mid = 0.5 * (lam_a + lam_b)
        f_mid = _f_lambda(G, Theta, lam_mid, msign_steps)
        if abs(f_mid) < best_f:
            best_lam, best_f = lam_mid, abs(f_mid)
        if abs(f_mid) <= tol:
            return lam_mid
        if f_mid < 0:
            lam_a, f_a = lam_mid, f_mid
        else:
            lam_b, f_b = lam_mid, f_mid

    return best_lam



def _target_radius(shape: Tuple[int, int], scaler: float = 1.0) -> float:
    """Compute spectral-muP target radius: scaler * sqrt(n_out / n_in)."""
    n_out, n_in = shape
    return scaler * math.sqrt(n_out / n_in)



class SSO(Optimizer):
    r"""Implements the Spectral Sphere Optimizer (SSO).

    SSO constrains weight matrices to lie on a spectral ball (the set of matrices
    whose largest singular value equals a target radius R) and performs Riemannian
    gradient descent on that manifold.

    SSO only handles **2-D weight matrices** (``nn.Linear`` weights). Pair it
    with :class:`torch.optim.AdamW` for 1-D parameters (biases, LayerNorm, etc.)
    using a parameter-group split.

    Args:
        params (iterable): Iterable of 2-D parameters or dicts defining
            parameter groups. All parameters must satisfy ``p.ndim == 2``.
        lr (float): Learning rate :math:`\eta` (default: ``0.02``).
        momentum (float): EMA coefficient :math:`\beta` for the momentum
            buffer (default: ``0.9``).
        nesterov (bool): Use Nesterov-style momentum (default: ``True``).
        radius_scaler (float): Multiplier for the spectral-muP target radius
            :math:`R = \text{scaler} \cdot \sqrt{n_\text{out}/n_\text{in}}`
            (default: ``1.0``).
        power_iter_steps (int): Number of bilateral power-iteration steps for
            estimating the leading singular value (default: ``5``).
        msign_steps (int): Number of Newton-Schulz iterations for the matrix
            sign computation (default: ``8``).
        solver_tol (float): Convergence tolerance for the :math:`\lambda^*`
            bisection solver (default: ``1e-6``).
        solver_max_iter (int): Maximum bisection iterations (default: ``20``).
        weight_decay (float): Decoupled weight-decay coefficient (default: ``0.0``).
        eps (float): Numerical stability epsilon (default: ``1e-7``).

    .. note::
        SSO is designed for 2-D parameters only. A typical usage pattern is::

            params_2d = [p for p in model.parameters() if p.ndim == 2]
            params_1d = [p for p in model.parameters() if p.ndim != 2]
            optimizer = ChainedOptimizer(
                SSO(params_2d, lr=0.02),
                torch.optim.AdamW(params_1d, lr=1e-3),
            )

    .. warning::
        This optimizer does **not** support ``foreach``, ``fused``, or
        ``capturable`` modes. It also does not support sparse gradients.

    Reference:
        Kosson et al., *Spectral Sphere Optimizer*, 2024.
        Aligned with Megatron-LM's ``SpectralBall`` implementation.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        momentum: float = 0.9,
        nesterov: bool = True,
        radius_scaler: float = 1.0,
        power_iter_steps: int = 5,
        msign_steps: int = 8,
        solver_tol: float = 1e-6,
        solver_max_iter: int = 20,
        weight_decay: float = 0.0,
        eps: float = 1e-7,
    ):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if power_iter_steps < 1:
            raise ValueError(f"power_iter_steps must be >= 1, got {power_iter_steps}")
        if msign_steps < 1:
            raise ValueError(f"msign_steps must be >= 1, got {msign_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            radius_scaler=radius_scaler,
            power_iter_steps=power_iter_steps,
            msign_steps=msign_steps,
            solver_tol=solver_tol,
            solver_max_iter=solver_max_iter,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

        # Validate: SSO only handles 2-D parameters
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"SSO only supports 2-D parameters, got shape {tuple(p.shape)}. "
                        "Use a separate optimizer (e.g. AdamW) for 1-D parameters."
                    )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", True)
            group.setdefault("weight_decay", 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            scaler: float = group["radius_scaler"]
            pi_steps: int = group["power_iter_steps"]
            ms_steps: int = group["msign_steps"]
            tol: float = group["solver_tol"]
            max_iter: int = group["solver_max_iter"]
            wd: float = group["weight_decay"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("SSO does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["prev_lambda"] = 0.0

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(grad, 1.0 - beta)

                # Nesterov: blend current grad toward the updated buffer
                M = grad.lerp(buf, beta) if nesterov else buf.clone()

                M_fp32 = M.float()
                M_norm = M_fp32 / M_fp32.norm().clamp_min(eps)

                sigma, u, v = _power_iteration(p, steps=pi_steps, eps=eps)

                R = _target_radius(p.shape, scaler)
                _apply_retract(p, sigma.item(), R, eps)
                Theta = u @ v.T  # shape: (n_out, n_in)
                lam_star = _solve_lambda(
                    M_norm, Theta,
                    init=state["prev_lambda"],
                    tol=tol,
                    max_iter=max_iter,
                    msign_steps=ms_steps,
                )
                state["prev_lambda"] = lam_star
                Phi = _msign(M_norm + lam_star * Theta, steps=ms_steps)

                p.add_(Phi.to(p.dtype), alpha=-lr * R)

        return loss
