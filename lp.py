import jax
import jax.numpy as jnp
import mpax

# def opt(t):
    
@jax.jit
def sol(t = 0):
    l = -jnp.array([jnp.inf,jnp.inf])
    u = jnp.array([jnp.inf,jnp.inf])
    # l = jnp.zeros((2,))
    # u = jnp.ones((2,))

    c = jnp.array([jnp.cos(t), jnp.sin(t)])
    A = jnp.zeros((0,2))
    b = jnp.array([])
    G = jnp.ones((1,2))
    h = jnp.array([0.5])

    lp = mpax.create_lp(c, A, b, G, h, l, u)
    solver = mpax.r2HPDHG(eps_abs=1e-4, eps_rel=1e-4, verbose=False)
    result = solver.optimize(lp)
    return result.primal_objective
