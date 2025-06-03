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


# import os
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=false "
#     "--xla_gpu_enable_async_collectives=true "
#     "--xla_gpu_enable_latency_hiding_scheduler=true "
#     "--xla_gpu_enable_highest_priority_async_stream=true "
# )


numit = 3000
batch = 10000
printevery = 100
cx = False
m,n,l,r = 2,2,2,7
# m,n,l,r = 3,3,3,23
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
        A = jnp.einsum('jl,kl->jkl', dec[1],dec[2]).reshape(b*c,r)
        B = jnp.einsum('kl,il->kil', dec[2],dec[0]).reshape(c*a,r)
        C = jnp.einsum('il,jl->ijl', dec[0],dec[1]).reshape(a*b,r)
        # solve = jax.vmap(lambda A,B: lx.linear_solve(lx.MatrixLinearOperator(A), B, lx.AutoLinearSolver(well_posed=None)).value, [None,1], 1)
        solve = jax.vmap(lambda A,B: lx.linear_solve(lx.MatrixLinearOperator(A.conj().T @ A, lx.positive_semidefinite_tag), 
                                                     A.conj().T @ B, lx.AutoLinearSolver(well_posed=True)).value, [None,1], 1)
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
        C = jnp.einsum('il,jl->ijl', dec[0],dec[1]).reshape(a*b,r)
        # solve = jax.vmap(lambda A,B: lx.linear_solve(lx.MatrixLinearOperator(A), B, lx.NormalCholesky()).value, [None,1], 1)
        solve = jax.vmap(lambda A,B: lx.linear_solve(lx.MatrixLinearOperator(A.conj().T @ A, lx.positive_semidefinite_tag), 
                                                     A.conj().T @ B, lx.AutoLinearSolver(well_posed=True)).value, [None,1], 1)
        # solve = jax.vmap(lambda A,B: lx.linear_solve(lx.MatrixLinearOperator(A), B, lx.SVD()).value, [None,1], 1)
        # solve = jax.vmap(lambda A,B: lx.linear_solve(lx.MatrixLinearOperator(A), B, lx.NormalCG(rtol=1e-2, atol=1e-3)).value, [None,1], 1)
        # solve = jax.vmap(lambda A,B: lx.linear_solve(lx.MatrixLinearOperator(A), B, lx.AutoLinearSolver(well_posed=None)).value, [None,1], 1)
        f3 = solve(C, T.reshape(a*b,c)).T
        return ((C @ f3.T).reshape(T.shape) - T,),dec+(f3,)
    return rs
    
def basic_loss_function(T):
    rs = residual_function(T)
    return lambda dec: jnp.sum(optax.l2_loss(jnp.abs(rs(dec)[0][0])))

def basic_loss_function_elim(T):
    rs = residual_function_elim(T)
    return lambda dec: jnp.sum(optax.l2_loss(jnp.abs(rs(dec)[0][0])))
    
cl_bound = 2
def clipped(x):
    if cx:
        return jnp.clip(jnp.real(x),min=-cl_bound,max=cl_bound) + jnp.clip(jnp.imag(x),min=-cl_bound,max=cl_bound) * 1j
    else:
        return jnp.clip(x,min=-cl_bound,max=cl_bound)
    
def loss_function(T):
    # rs = residual_function_elim(T)
    rs = residual_function(T)
    # rs = residual_function_fp(T)
    def loss_fn(dec,it,rng):
        progress = it / numit
        r,full_dec = rs(dec)
        base_loss = jax.tree.reduce(jnp.add,jax.tree.map(lambda e: 5*jnp.mean(optax.l2_loss(jnp.abs(e))), r))
        regularization_loss = 0.0
        descretization_loss = 0.0
        for f in full_dec:
            regularization_loss += 0.02 * jnp.where(progress < 1/3, 0.1 ** (progress*6), 0) * jnp.mean(optax.l2_loss(jnp.abs(f)))
            regularization_loss += 0.1 * jnp.mean(optax.l2_loss(jnp.abs(f-clipped(f))))
            # descretization_loss += 0.003 * jnp.mean(jnp.abs(f))
            # descretization_loss += ((1-jnp.cos(jnp.pi*progress))/2) * 0.03 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f))))
            descretization_loss += ((1-jnp.cos(2*jnp.pi*progress))/2) * 0.02 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f))))
            # descretization_loss += ((1-jnp.cos(jnp.pi*progress))/2) * 0.04 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f*2)/2)))
            # loss += 0.03 * ((1-jnp.cos(jnp.pi*progress))/2) * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f))))
            # loss += 0.01 * ((1-jnp.cos(jnp.pi*progress))/2)**2 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f*2)/2)))
            # loss += 0.1 * jnp.mean(optax.l2_loss(jnp.abs(f-clipped(f))))
        loss = base_loss + regularization_loss + descretization_loss
        return loss,(full_dec, { (0,'base_loss') : base_loss, 
                                (1,'regularization_loss') : regularization_loss, 
                                (2,'descretization_loss') : descretization_loss })
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
basic_loss_fn = jax.vmap(basic_loss_function(T))

rng = jax.random.PRNGKey(np.random.randint(2**32))

rng, rngc = jax.random.split(rng)
dec = jax.vmap(init)(jax.random.split(rngc,batch))

start_learning_rate = 0.1
optimizer = optax.adam(start_learning_rate)
# optimizer = optax.adamw(start_learning_rate)
# optimizer = optax.lbfgs(start_learning_rate)
opt_state = jax.vmap(optimizer.init)(dec)

@functools.partial(jax.jit,donate_argnums=[0,1])
@functools.partial(jax.vmap, in_axes=[0,0,None,None])
def update_function(dec, opt_state, it, rng):
    (loss, (full_dec, losses)), grads = jax.value_and_grad(loss_fn,has_aux=True)(dec,it,rng)
    if cx:
        grads = jax.tree.map(jnp.conj, grads)
    # grads = jax.tree.map(jax.add, grads, 
    #                      jax.random.normal(rng, jax.tree.structure(
    # jax.random.normal(rng, jax.tree.structure(grads)
    # grads = jax.tree.map(lambda e: e + 0.02*jax.random.normal(rng, e.shape, e.dtype),grads)
    updates, opt_state = optimizer.update(grads, opt_state, dec, value = loss, grad = grads, value_fn = lambda x: loss_fn(x,it,rng))
    dec = optax.apply_updates(dec, updates)
    return dec, opt_state, loss, full_dec, losses

@jax.jit
def stats(opt_state, loss, full_dec, losses):
    besti = jnp.argpartition(loss,4)[:5]
    besti = besti[jnp.argsort(loss[besti])]
    # besti = jnp.sort(besti)
    rfd = jax.tree.map(lambda e: e[besti], full_dec)
    bloss = basic_loss_fn(rfd)
    maxabs = jax.vmap(lambda dec: jax.tree.reduce(jnp.maximum,jax.tree.map(lambda e: jnp.max(jnp.abs(e)), dec)))(rfd)
    worst = jnp.max(loss)
    res = { (0, '') : besti,
           (1, 'loss') : 1000*loss[besti]}
    for (i,s), v in jax.tree.map(lambda e: 1000*e[besti], losses).items():
        res[(2+i,s)] = v
    res[(2+len(losses), 'squared l2 norm')] = bloss
    res[(3+len(losses), 'l-oo norm')] = maxabs
    res[(4+len(losses), 'worst batch loss')] = 1000*worst
    return res

# A simple update loop.
for it in range(numit):
    if it == 1:
        startt = time.time()
    rng, rngcur = jax.random.split(rng)
    dec, opt_state, loss, full_dec, losses = update_function(dec, opt_state, it, rngcur)
    if it % printevery == printevery-1:
        sts = stats(opt_state,loss,full_dec,losses)
        elapsed = time.time() - startt
        print(f'{it}: {elapsed:.3f}s elapsed, {it*batch / (1000*elapsed):.0f}K iteratons/s')
        for (_,desc), v in sorted(sts.items()):
            print(desc.rjust(25), v)
    else:
        del loss, full_dec, losses
        
bloss_round = basic_loss_fn(jax.tree.map(lambda x: jnp.round(x),dec))
# bloss_round = basic_loss_fn(jax.tree.map(lambda x: jnp.round(x*2)/2,dec))
successes = jnp.count_nonzero(bloss_round == 0)
print( f"{successes} solves out of {batch}, {successes/batch*100:.2f}%" )

startt = time.time()
refine_batch = 1000
numrefine = batch
numrefine = min(numrefine,batch)
loss = basic_loss_fn(dec)
besti = jnp.argpartition(loss,numrefine-1)[:numrefine]
decbest = jax.tree.map(lambda x: x[besti],dec)
@jax.jit
# @jax.vmap
def refine(dec):
    # lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4,verbose=frozenset(['loss']))
    lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4,linear_solver=lx.NormalCholesky())
    # lm = optx.Newton(rtol=1e-3, atol=1e-4, cauchy_termination=False)
    rs_base = residual_function(T)
    # # lm = optx.Newton(rtol=1e-3, atol=1e-4, linear_solver=lx.NormalCG(rtol=1e-2,atol=1e-3), cauchy_termination=False)
    # lm = optx.Newton(rtol=1e-3, atol=1e-4, linear_solver=lx.AutoLinearSolver(well_posed=True), cauchy_termination=False)
    # rs_base = residual_function_fp(T)
    def rs_fn(dec, unused):
        rs, fdec = rs_base(dec)
        rs = [rs]
        for f in dec:
            rs.append(0.5 * (f - jnp.round(f)))
            # rs.append(0.5 * (f - jnp.round(f*2)/2))
        #     # rs.append(0.1 * (f - jnp.round(f)))
        #     rs.append(0.5 * (f - clipped(f)))
        return rs
    # return optx.root_find(rs_fn, lm, dec, throw=False).value
    return optx.least_squares(rs_fn, lm, dec, throw=False).value
# dec_refine = refine(decbest)
dec_refine = jax.lax.map(refine, decbest, batch_size=refine_batch)
bloss = basic_loss_fn(dec_refine) 
besti = jnp.argpartition(bloss,9)[:10]
print( bloss[besti] )
bloss_round = basic_loss_fn(jax.tree.map(lambda x: jnp.round(x),dec_refine))
# bloss_round = basic_loss_fn(jax.tree.map(lambda x: jnp.round(x*2)/2,dec_refine))
print( bloss_round[besti] )
successes = jnp.count_nonzero(bloss_round == 0)
print( f"{successes} solves out of {numrefine}, {successes/numrefine*100:.2f}%" )
time_elapsed = time.time() - startt
print(f'{time_elapsed:.3f}s elapsed, {numrefine/time_elapsed:.3f} solves/s')
