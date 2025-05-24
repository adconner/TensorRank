import numpy as np
import jax.numpy as jnp
import jax
import optax
import functools
import optimistix as optx
import lineax as lx
from itertools import *

numit = 30000
cx = False
m,n,l,r = 2,2,2,7
# m,n,l,r = 3,3,3,23
# m,n,l,r = 4,4,4,48
batch = 10000
# batch = 10000

import time
startt = time.time()

def matrixmult(m,n,l):
    T=np.zeros((m*n,n*l,l*m))
    for i,j,k in product(range(m),range(n),range(l)):
        T[i*n+j, j*l+k, k*m+i] = 1
    return T

def residual_function(T):
    def rs(dec):
        S = jnp.einsum('il,jl,kl->ijk', *dec)
        return (S-T,)
    return rs

def residual_function_fp(T):
    def rs(dec):
        r = dec[0].shape[1]
        a,b,c = [f.shape[0] for f in dec]
        A = lx.MatrixLinearOperator(jnp.einsum('jl,kl->jkl', *dec).reshape(b*c,r))
        B = lx.MatrixLinearOperator(jnp.einsum('kl,il->kil', *dec).reshape(c*a,r))
        C = lx.MatrixLinearOperator(jnp.einsum('il,jl->ijl', *dec).reshape(a*b,r))
        f1 = lx.linear_solve(A, T.transpose(1,2,0).reshape(b*c,a), lx.AutoLinearSolver(well_posed=None)).T
        f2 = lx.linear_solve(B, T.transpose(2,0,1).reshape(c*a,b), lx.AutoLinearSolver(well_posed=None)).T
        f3 = lx.linear_solve(C, T.reshape(a*b,c), lx.AutoLinearSolver(well_posed=None)).T
        return (dec[0] - f1, dec[1] - f2, dec[2] - f3)
    return rs

def l2_squared(x):
    def f1(x):
        if cx:
            return jnp.sum(jnp.real(x.conj() * x))
        else:
            return jnp.sum(x * x)
    return jax.tree.reduce(jnp.add, jax.tree.map(f1, x))
    
def basic_loss_function(T):
    rs = residual_function(T)
    return lambda dec: l2_squared(rs(dec))
    
cl_bound = 2
def clipped(x):
    if cx:
        return jnp.clip(jnp.real(x),min=-cl_bound,max=cl_bound) + jnp.clip(jnp.imag(x),min=-cl_bound,max=cl_bound) * 1j
    else:
        return jnp.clip(x,min=-cl_bound,max=cl_bound)
    
def loss_function(T):
    rs = residual_function(T)
    def loss_fn(dec,it,rng):
        progress = it / numit
        r = rs(dec)
        # sched = jnp.maximum(0, 3*progress-2)**2 / 2
        # loss =  (1-sched) * l2_squared(r) + sched * jnp.sum(jnp.abs(r))
        loss = l2_squared(r)
        for f in dec:
            # loss += 0.01 * jnp.sum(jnp.abs(f - jnp.round(f)))
            loss += 0.02 * progress * jnp.sum(jnp.abs(f - jnp.round(f*2)/2))
            loss += 0.1 * progress * jnp.sum(l2_squared(f - clipped(f))) +\
                    0.1 * (1-progress) * l2_squared(f)
                    # 0.1 * (1-progress) * jnp.sum(jnp.abs(f))
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
# optimizer = optax.adam(start_learning_rate)
optimizer = optax.adamw(start_learning_rate)
# optimizer = optax.lbfgs(start_learning_rate)
opt_state = jax.vmap(optimizer.init)(dec)

@jax.jit
@functools.partial(jax.vmap, in_axes=[0,0,None,None])
def update_function(dec, opt_state, it, rng):
    loss, grads = jax.value_and_grad(loss_fn)(dec,it,rng)
    if cx:
        grads = jax.tree.map(jnp.conj, grads)
    updates, opt_state = optimizer.update(grads, opt_state, dec, value = loss, grad = grads, value_fn = lambda x: loss_fn(x,it,rng))
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
startt = time.time()

with jax.default_device(jax.devices('cpu')[0]):
    numrefine = 200
    besti = jnp.argpartition(loss,numrefine)[:numrefine]
    decbest = jax.tree.map(lambda x: x[besti],dec)
    @jax.jit
    @jax.vmap
    def refine(dec):
        # lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4,verbose=frozenset(['loss']))
        lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4)
        rs_base = residual_function(T)
        @jax.jit
        def rs_fn(dec, unused):
            rs = [rs_base(dec)]
            # for f in dec:
            #     rs.append(0.1 * (f - jnp.round(f*2)/2))
            #     # rs.append(0.1 * (f - jnp.round(f)))
            #     rs.append(0.5 * (f - clipped(f)))
            return rs
        return optx.least_squares(rs_fn, lm, dec).value
    dec_refine = refine(decbest)
basic_loss_fn = jax.jit(jax.vmap(basic_loss_function(T)),device=jax.devices('cpu')[0])
print( basic_loss_fn(dec_refine) )
print(f'time elapsed : {time.time() - startt}')
