import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize

from .helpers import truncate_by_g, mse, cross_entropy, truncate_all_by_g
from .att import att_estimates


## (Courtesy of Shi et al.)

def _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps):
    """
    Helper for psi_tmle_bin_outcome

    Returns q_\eps (t,x)
    (i.e., value of perturbed predictor at t, eps, x; where q_t0, q_t1, g are all evaluated at x
    """
    h = t * (1./g) - (1.-t) / (1. - g)
    full_lq = (1.-t)*logit(q_t0) + t*logit(q_t1)  # logit predictions from unperturbed model
    logit_perturb = full_lq + eps * h
    return expit(logit_perturb)


def psi_tmle_bin_outcome(q_t0, q_t1, g, t, y, truncate_level=0.05):
    # TODO: make me useable
    # solve the perturbation problem

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    eps_hat = minimize(lambda eps: cross_entropy(y, _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps))
                       , 0., method='Nelder-Mead')

    eps_hat = eps_hat.x[0]

    def q1(t_cf):
        return _perturbed_model_bin_outcome(q_t0, q_t1, g, t_cf, eps_hat)

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    return np.mean(ite)


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05, eps=1e-8):
    # Ensure flat float arrays
    q_t0 = np.asarray(q_t0, dtype=float).reshape(-1)
    q_t1 = np.asarray(q_t1, dtype=float).reshape(-1)
    g    = np.asarray(g,    dtype=float).reshape(-1)
    t    = np.asarray(t,    dtype=float).reshape(-1)
    y    = np.asarray(y,    dtype=float).reshape(-1)

    # Optionally drop/extreme-propensity rows; your helper presumably does this
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    n = t.shape[0]
    if n == 0:
        raise ValueError("No data left after truncation.")

    # Clip to avoid division nasties even after truncation
    g = np.clip(g, eps, 1.0 - eps)

    # Diagnostics for g (MSE vs binary labels)
    g_loss = np.mean((g - t)**2)
    print(f"g_loss: {g_loss}")

    # Clever covariate & initial Q
    h = t / g - (1.0 - t) / (1.0 - g)
    Q0 = q_t0
    Q1 = q_t1
    Q  = (1.0 - t) * Q0 + t * Q1  # unfluctuated prediction at observed t

    # One-dimensional fluctuation with identity link
    denom = np.sum(h * h)
    if eps_hat is None:
        if denom < 1e-12:
            eps_hat = 0.0  # degenerate case; no fluctuation
        else:
            eps_hat = np.sum(h * (y - Q)) / denom

    def Q_star(t_cf):
        t_cf = np.asarray(t_cf, dtype=float).reshape(-1)
        h_cf = t_cf / g - (1.0 - t_cf) / (1.0 - g)
        Q_cf = (1.0 - t_cf) * Q0 + t_cf * Q1
        return Q_cf + eps_hat * h_cf

    # ATE: mean of Q*(1, X) - Q*(0, X)
    ITE_star = Q_star(np.ones_like(t)) - Q_star(np.zeros_like(t))
    psi_tmle = np.mean(ITE_star)

    # Influence curve: (Q*(1)-Q*(0) - psi) + h * (y - Q*(t))
    IC = (ITE_star - psi_tmle) + h * (y - Q_star(t))

    # Finite-sample centered SE with sample var
    ICc = IC - IC.mean()
    psi_tmle_std = ICc.std(ddof=1) / np.sqrt(n)

    initial_loss = np.mean((Q - y)**2)
    final_loss   = np.mean((Q_star(t) - y)**2)

    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss


def psi_iptw(q_t0, q_t1, g, t, y, truncate_level=0.05):
    ite=(t / g - (1-t) / (1-g))*y
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_aiptw(q_t0, q_t1, g, t, y, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    full_q = q_t0 * (1 - t) + q_t1 * t
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    ite = h * (y - full_q) + q_t1 - q_t0

    return np.mean(ite)

def psi_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_very_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    return y[t == 1].mean() - y[t == 0].mean()


def ates_from_atts(q_t0, q_t1, g, t, y, truncate_level=0.05):
    """
    Sanity check code: ATE = ATT_1*P(T=1) + ATT_0*P(T=1)

    :param q_t0:
    :param q_t1:
    :param g:
    :param t:
    :param y:
    :param truncate_level:
    :return:
    """

    prob_t = t.mean()

    att = att_estimates(q_t0, q_t1, g, t, y, prob_t, truncate_level=truncate_level)
    atnott = att_estimates(q_t1, q_t0, 1.-g, 1-t, y, 1.-prob_t, truncate_level=truncate_level)

    att['one_step_tmle'] = att['one_step_tmle'][0]
    atnott['one_step_tmle'] = atnott['one_step_tmle'][0]

    ates = {}
    for k in att.keys():
        ates[k] = att[k]*prob_t + atnott[k]*(1.-prob_t)

    return ates

def main():
    pass


if __name__ == "__main__":
    main()