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
    def rs(dec):
        S = jnp.einsum('il,jl,kl->ijk', *dec)
        return S-T
    return rs

numit = 10000
cx = True
m,n,l,r = 4,4,4,48
# m,n,l,r = 4,4,4,48
batch = 20000

cl_bound = 2
def clipped(x):
    if cx:
        return jnp.clip(jnp.real(x),min=-cl_bound,max=cl_bound) + jnp.clip(jnp.imag(x),min=-cl_bound,max=cl_bound) * 1j
    else:
        return jnp.clip(x,min=-cl_bound,max=cl_bound)
def loss_function(T):
    rs = residual_function(T)
    def l2_squared(x):
        if cx:
            return jnp.sum(jnp.real(x.conj() * x))
        else:
            return jnp.sum(x * x)
    def loss_fn(dec,it,rng):
        r = rs(dec)
        loss = l2_squared(r)
        for f in dec:
            # loss += 0.01 * jnp.sum(jnp.abs(f - jnp.round(f)))
            loss += 0.01 * jnp.sum(jnp.abs(f - jnp.round(f*2)/2))
            loss += 0.1 * jnp.sum(jnp.abs(f - clipped(f)))
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
    if it % 1000 == 999:
        besti = jnp.argpartition(loss,4)[:5]
        besti = besti[jnp.argsort(loss[besti])]
        # besti = jnp.sort(besti)
        print(it)
        print(besti)
        print(loss[besti])
        print(maxabs[besti])
        print(jnp.max(loss))
        # print(loss)
        
print(f'time elapsed : {time.time() - startt}')

with jax.default_device(jax.devices('cpu')[0]):
    numrefine = 5
    decs_refined = []
    decrs = []
    besti = jnp.argpartition(loss,numrefine)[:numrefine]
    for i in besti:
        curdec = jax.tree.map(lambda f: f[i],dec)
        lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4, verbose=frozenset(['loss','step_size']))
        rs_base = residual_function(T)
        @jax.jit
        def rs_fn(dec, unused):
            rs = [rs_base(dec)]
            for f in dec:
                rs.append(0.1 * (f - jnp.round(f*2)/2))
                # rs.append(0.1 * (f - jnp.round(f)))
                rs.append(0.5 * (f - clipped(f)))
            return rs
        res = optx.least_squares(rs_fn, lm, curdec)
        curdec = res.value
        curdecr = jax.tree.map(lambda x: jnp.round(x*2)/2, curdec)
        decs_refined.append(curdec)
        decrs.append(curdecr)

