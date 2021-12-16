import s4
import jax.numpy as np

def test_ssm():
    L = 16
    s = s4.S4(L, 2)
    out = s4.K_conv_naive(s.discrete, L)


    out2 = s4.K_gen_naive(s.discrete, L)
    out2 = s4.convFromGen(out2, L)
    assert np.allclose(out, out2, atol=1e-2, rtol=1e-2)

    out3 = s4.K_gen_inverse(s.discrete, L)
    out3 = s4.convFromGen(out3, L)
    assert np.allclose(out2, out3, atol=1e-2, rtol=1e-2)

    out4 = s.K_gen()
    out4 = s4.convFromGen(out4, L)
    
    assert np.allclose(out2, out4, atol=1e-2, rtol=1e-2)
