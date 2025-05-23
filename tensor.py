import numpy as np
import jax.numpy as jnp
import jax
import optax
import functools
import optimistix as optx
from itertools import *

import time
startt = time.time()

def matrixmult(m,n,l):
    T=np.zeros((m*n,n*l,l*m))
    for i,j,k in product(range(m),range(n),range(l)):
        T[i*n+j, j*l+k, k*m+i] = 1
    return T

def complex_l2_loss(x,y):
    return jnp.real((x-y).conj() * (x-y))

def residual_function(T):
    def rs(dec,unused):
        S = jnp.einsum('il,jl,kl->ijk', *dec)
        return S-T
    return rs

numit = 10000
cx = False
m,n,l,r = 2,2,2,7
batch = 10000

def loss_function(T):
    rs = residual_function(T)
    def l2_squared(x):
        if cx:
            return jnp.sum(jnp.real(x.conj() * x))
        else:
            return jnp.sum(x * x)
    def loss_fn(dec,it,rng):
        r = rs(dec,None)
        loss = l2_squared(r)
        return loss
    return loss_fn

def init(rng):
    dec = []
    for td in T.shape:
        rng, rngcur = jax.random.split(rng)
        if cx:
            dec.append(jax.random.normal(rng, (td, r), jnp.complex64))
        else:
            dec.append(jax.random.normal(rng, (td, r)))
    dec = tuple(dec)
    dec = jax.tree.map(lambda e: e/5, dec)
    return dec

T = matrixmult(m,n,l)
loss_fn = loss_function(T)

rng = jax.random.PRNGKey(np.random.randint(2**32))

rng, rngc = jax.random.split(rng)
dec = jax.vmap(init)(jax.random.split(rngc,batch))

start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
# optimizer = optax.adamw(start_learning_rate)
# optimizer = optax.lbfgs(start_learning_rate)
opt_state = jax.vmap(optimizer.init)(dec)

@jax.jit
@functools.partial(jax.vmap, in_axes=[0,0,None,None])
def update_function(dec, opt_state, it, rng):
    loss, grads = jax.value_and_grad(loss_fn)(dec,it,rng)
    if cx:
        grads = jax.tree.map(jnp.conj, grads)
    updates, opt_state = optimizer.update(grads, opt_state, dec, value = loss, grad = grads, value_fn = loss_fn)
    dec = optax.apply_updates(dec, updates)
    maxabs = jax.tree.reduce(jnp.maximum,jax.tree.map(lambda t: jnp.max(jnp.abs(t)),dec))
    return dec, opt_state, loss, maxabs

# A simple update loop.
for it in range(numit):
    rng, rngcur = jax.random.split(rng)
    dec, opt_state, loss, maxabs = update_function(dec, opt_state, it, rngcur)
    besti = jnp.argpartition(loss,4)[:5]
    besti = besti[jnp.argsort(loss[besti])]
    # besti = jnp.sort(besti)
    if it % 100 == 0:
        print(it)
        print(besti)
        print(loss[besti])
        print(maxabs[besti])
        print(jnp.max(loss))
        # print(loss)

print(f'time elapsed : {time.time() - startt}')

# lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4, verbose=frozenset(['loss','step_size']))
# rs_fn = jax.jit(residual_function(T))
# res = optx.least_squares(rs_fn, lm, dec)

