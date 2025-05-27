import numpy as np
import jax.numpy as jnp
import jax
import optax
import functools
import optimistix as optx
import lineax as lx
import equinox as eqx
from itertools import *
import time

numit = 100000
batch = 1000
printevery = 1000
cx = False
# m,n,l,r = 2,2,2,7
m,n,l,r = 3,3,3,23
# m,n,l,r = 4,4,4,48

def matrixmult(m,n,l):
    T=np.zeros((m*n,n*l,l*m))
    for i,j,k in product(range(m),range(n),range(l)):
        T[i*n+j, j*l+k, k*m+i] = 1
    return T

def residual_function(T):
    def rs(dec):
        S = jnp.einsum('il,jl,kl->ijk', *dec)
        return (S-T,),dec
    return rs

def residual_function_fp(T):
    def rs(dec):
        r = dec[0].shape[1]
        a,b,c = T.shape
        A = lx.MatrixLinearOperator(jnp.einsum('jl,kl->jkl', dec[1],dec[2]).reshape(b*c,r))
        B = lx.MatrixLinearOperator(jnp.einsum('kl,il->kil', dec[2],dec[0]).reshape(c*a,r))
        C = lx.MatrixLinearOperator(jnp.einsum('il,jl->ijl', dec[0],dec[1]).reshape(a*b,r))
        solve = jax.vmap(lambda A,B: lx.linear_solve(A, B, lx.AutoLinearSolver(well_posed=None)).value, [None,1], 1)
        f1 = solve(A, T.transpose(1,2,0).reshape(b*c,a)).T
        f2 = solve(B, T.transpose(2,0,1).reshape(c*a,b)).T
        f3 = solve(C, T.transpose(0,1,2).reshape(a*b,c)).T
        return (dec[0] - f1, dec[1] - f2, dec[2] - f3),dec
    return rs

def residual_function_elim(T):
    def rs(dec):
        assert len(dec) == 2
        r = dec[0].shape[1]
        a,b,c = T.shape
        C = lx.MatrixLinearOperator(jnp.einsum('il,jl->ijl', dec[0],dec[1]).reshape(a*b,r))
        solve = jax.vmap(lambda A,B: lx.linear_solve(A, B, lx.AutoLinearSolver(well_posed=None)).value, [None,1], 1)
        f3 = solve(C, T.reshape(a*b,c)).T
        return (jax.vmap(C.mv, out_axes=1)(f3).reshape(T.shape) - T,),dec+(f3,)
    return rs
    
def basic_loss_function(T):
    rs = residual_function(T)
    return lambda dec: jnp.sum(optax.l2_loss(jnp.abs(rs(dec)[0][0])))

def basic_loss_function_elim(T):
    rs = residual_function_elim(T)
    return lambda dec: jnp.sum(optax.l2_loss(jnp.abs(rs(dec)[0][0])))
    
cl_bound = 1
def clipped(x):
    if cx:
        return jnp.clip(jnp.real(x),min=-cl_bound,max=cl_bound) + jnp.clip(jnp.imag(x),min=-cl_bound,max=cl_bound) * 1j
    else:
        return jnp.clip(x,min=-cl_bound,max=cl_bound)
    
def loss_function(T):
    # rs = residual_function_elim(T)
    rs = residual_function(T)
    def loss_fn(dec,it,rng):
        progress = it / numit
        r,dec = rs(dec)
        loss = jax.tree.reduce(jnp.add,jax.tree.map(lambda e: jnp.mean(optax.l2_loss(jnp.abs(e))), r))
        for f in dec:
            loss += 0.003 * 0.1 ** (progress*4) * jnp.mean(jnp.abs(f))
            loss += 0.01 * 0.1 ** (progress*4) * jnp.mean(optax.l2_loss(jnp.abs(f)))
            # loss += 0.03 * ((1-jnp.cos(jnp.pi*progress))/2) * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f))))
            # loss += 0.01 * ((1-jnp.cos(jnp.pi*progress))/2)**2 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f*2)/2)))
            # loss += 0.1 * jnp.mean(optax.l2_loss(jnp.abs(f-clipped(f))))
        return 1000*loss
    return loss_fn

def init(rng):
    dec = []
    for td in T.shape:
    # for td in T.shape[:-1]:
        rng, rngcur = jax.random.split(rng)
        if cx:
            dec.append(jax.random.normal(rng, (td, r), jnp.complex64))
        else:
            dec.append(jax.random.normal(rng, (td, r)))
    dec = tuple(dec)
    dec = jax.tree.map(lambda e: e/5, dec)
    return dec

T = matrixmult(m,n,l)
if cx:
    T = T.astype('complex64')
loss_fn = loss_function(T)
basic_loss_fn = jax.jit(jax.vmap(basic_loss_function(T)),device=jax.devices('cpu')[0])
# basic_loss_fn = jax.jit(jax.vmap(basic_loss_function_elim(T)),device=jax.devices('cpu')[0])

rng = jax.random.PRNGKey(np.random.randint(2**32))

rng, rngc = jax.random.split(rng)
dec = jax.vmap(init)(jax.random.split(rngc,batch))

start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
# optimizer = optax.adamw(start_learning_rate)
# optimizer = optax.lbfgs(start_learning_rate)
opt_state = jax.vmap(optimizer.init)(dec)

@functools.partial(jax.jit,donate_argnums=[0,1])
@functools.partial(jax.vmap, in_axes=[0,0,None,None])
def update_function(dec, opt_state, it, rng):
    loss, grads = jax.value_and_grad(loss_fn)(dec,it,rng)
    if cx:
        grads = jax.tree.map(jnp.conj, grads)
    # grads = jax.tree.map(jax.add, grads, 
    #                      jax.random.normal(rng, jax.tree.structure(
    # jax.random.normal(rng, jax.tree.structure(grads)
    # grads = jax.tree.map(lambda e: e + 0.02*jax.random.normal(rng, e.shape, e.dtype),grads)
    updates, opt_state = optimizer.update(grads, opt_state, dec, value = loss, grad = grads, value_fn = lambda x: loss_fn(x,it,rng))
    dec = optax.apply_updates(dec, updates)
    maxabs = jax.tree.reduce(jnp.maximum,jax.tree.map(lambda t: jnp.max(jnp.abs(t)),dec))
    return dec, opt_state, loss, maxabs

# A simple update loop.
for it in range(numit):
    if it == 1:
        startt = time.time()
    rng, rngcur = jax.random.split(rng)
    dec, opt_state, loss, maxabs = update_function(dec, opt_state, it, rngcur)
    # if it % 10 == 9:
    if it % printevery == printevery-1:
        besti = jnp.argpartition(loss,4)[:5]
        besti = besti[jnp.argsort(loss[besti])]
        # besti = jnp.sort(besti)
        elapsed = time.time() - startt
        print(f'{it}: time elapsed {elapsed}, Kips {it*batch / (1000*elapsed) }')
        print(besti)
        print(loss[besti])
        print(basic_loss_fn(dec)[besti])
        print(maxabs[besti])
        print(jnp.max(loss))
        # print(loss)
        
# startt = time.time()
# numrefine = min(500,batch)
# loss = basic_loss_fn(dec)
# besti = jnp.argpartition(loss,numrefine-1)[:numrefine]
# decbest = jax.tree.map(lambda x: x[besti],dec)
# # @jax.jit
# # @eqx.filter_jit
# @jax.vmap
# def refine(dec):
#     # lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4,verbose=frozenset(['loss']))
#     lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4)
#     # lm = optx.Newton(rtol=1e-3, atol=1e-4, cauchy_termination=False)
#     rs_base = residual_function(T)
#     # # lm = optx.Newton(rtol=1e-3, atol=1e-4, linear_solver=lx.NormalCG(rtol=1e-2,atol=1e-3), cauchy_termination=False)
#     # lm = optx.Newton(rtol=1e-3, atol=1e-4, linear_solver=lx.AutoLinearSolver(well_posed=True), cauchy_termination=False)
#     # rs_base = residual_function_fp(T)
#     @jax.jit
#     def rs_fn(dec, unused):
#         rs = rs_base(dec)
#         # for f in dec:
#         #     rs.append(0.1 * (f - jnp.round(f*2)/2))
#         #     # rs.append(0.1 * (f - jnp.round(f)))
#         #     rs.append(0.5 * (f - clipped(f)))
#         return rs
#     return optx.root_find(rs_fn, lm, dec, throw=False).value
#     # return optx.least_squares(rs_fn, lm, dec, throw=False).value
# dec_refine = refine(decbest)
# print( basic_loss_fn(dec_refine) )
# print(f'time elapsed : {time.time() - startt}')
