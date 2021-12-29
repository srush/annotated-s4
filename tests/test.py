import jax
import jax.numpy as np
from jax.numpy.linalg import eig, inv, matrix_power
import s4


def test_ssm():
    N = 4
    L = 8
    I = np.eye(4)
    rng = jax.random.PRNGKey(0)
    # s = s4.S4(L, 2)
    A = s4.make_HiPPO(N)
    B = jax.random.uniform(rng, (N, 1))
    C = jax.random.uniform(rng, (1, N))

    ssm = s4.discretize(A, B, C, 1.0 / L)
    out = s4.K_conv(*ssm, L)

    out2 = s4.K_gen_simple(*ssm, L=L)
    out2 = s4.convFromGen(out2, L)
    assert np.allclose(out, out2, atol=1e-2, rtol=1e-2)

    out3 = s4.K_gen_inverse(*ssm, L=L)
    out3 = s4.convFromGen(out3, L)
    assert np.allclose(out2, out3, atol=1e-2, rtol=1e-2)

    u = jax.random.uniform(rng, (L,))
    y2 = s4.scanSSM(
        s4.stepSSM(*ssm), u[:, np.newaxis], np.zeros((ssm[0].shape[0],))
    ).ravel()
    y3 = s4.nonCircularConvolution(u, out)
    assert np.allclose(y2, y3, atol=1e-2, rtol=1e-2)


def test_s4():
    N = 4
    L = 8
    I = np.eye(4)

    A2, gamma, p, q, V = s4.make_DPLR_HiPPO(N)
    Vc = V.conj().T
    A = np.diag(gamma) - p[:, np.newaxis] * q[np.newaxis, :].conj()
    A3 = (
        V
        @ (
            np.diag(gamma)
            - (Vc @ p[:, np.newaxis]) @ (Vc @ q[:, np.newaxis].conj()).conj().T
        )
        @ Vc
    )
    A4 = V @ np.diag(gamma) @ Vc - (p[:, np.newaxis] @ q[np.newaxis, :])
    assert np.allclose(A2, A3, atol=1e-2, rtol=1e-2)
    assert np.allclose(A2, A4, atol=1e-2, rtol=1e-2)

    rng = jax.random.PRNGKey(0)
    B = jax.random.uniform(rng, (N, 1))
    C = jax.random.uniform(rng, (1, N))

    step = 1.0 / L
    ssm = s4.discretize(A, B, C, step)

    Abar, _, Cbar = ssm
    Ct = (I - matrix_power(Abar, L)).conj().T @ Cbar.ravel()

    out2 = s4.K_gen_simple(*ssm, L=L)
    out2 = s4.convFromGen(out2, L)

    K_gen = s4.K_gen_DPLR(gamma, p, q, B, Ct, step)
    out4 = s4.convFromGen(K_gen, L)
    assert np.allclose(out2, out4, atol=1e-2, rtol=1e-2)

    # out4 = s.K_gen()
    # out4 = s4.convFromGen(out4, L)

    # assert np.allclose(out2, out4, atol=1e-2, rtol=1e-2)
