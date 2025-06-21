import numpy as np
import jax.numpy as jnp
import jax
import optax
import functools
from itertools import *
import time
import scipy

numit = 10000
batch = 100
printevery = 100
cx = False
# m,n,l,r = 2,2,2,7
m,n,l,r = 3,3,3,20
# m,n,l,r = 4,4,4,48

print_num = 5
print_num = min(print_num,batch)

def matrixmult(m,n,l):
    T=np.zeros((m*n,n*l,l*m))
    for i,j,k in product(range(m),range(n),range(l)):
        T[i*n+j, j*l+k, k*m+i] = 1
    return T
    
T = matrixmult(m,n,l)
if cx:
    T = T.astype('complex64')
    
def tight_weight(T):
    from sage.all import matrix,ZZ
    a,b,c = T.shape
    I,J,K = np.nonzero(T)
    eqs = []
    for i,j,k in zip(I,J,K):
        eqs.append([1 if e == i else 0 for e in range(a)] +\
            [1 if e == j else 0 for e in range(b)] +\
            [1 if e == k else 0 for e in range(c)])
    tights = matrix(ZZ,eqs).right_kernel_matrix(basis='LLL')
    # tights = np.array(tights[:,:a]), np.array(tights[:,a:a+b]), np.array(tights[:,a+b:])
    # return tights
    tights = [(t[:a],t[a:a+b],t[a+b:]) for t in tights]
    M = matrix([[ta[i]+tb[j]+tc[k] for i,j,k in
        product(range(a),range(b),range(c))] for ta,tb,tc in tights])
    return np.array(M).reshape(len(tights),a,b,c)

tight = tight_weight(T)

def init(key):
    dshape = (tuple(jnp.zeros((d,r),jnp.complex64 if cx else jnp.float32) 
                    for d in T.shape), jnp.zeros(tight.shape[0]))
    return optax.tree.random_like(key, dshape, jax.random.normal)

key = jax.random.PRNGKey(np.random.randint(2**63))
key, rngc = jax.random.split(key)
dect = jax.vmap(init)(jax.random.split(rngc,batch))

@jax.jit
def basic_loss(dect,temp):
    dec, t = dect
    S = jnp.einsum('il,jl,kl->ijk', *dec)
    t = jnp.where( temp == 0.0, jax.lax.stop_gradient(t), t)
    tightexp = jnp.einsum('i,ijkl->jkl', t, tight)
    influence = jnp.where( temp == 0.0, 
                          jnp.where( tightexp <= 0.0, 1.0, 0.0 ),
                          jax.nn.sigmoid(-tightexp/temp) )
    E = (S-T) * influence
    return jnp.mean(optax.l2_loss(E))

cl_bound = 2
def clipped(x):
    if cx:
        return jnp.clip(jnp.real(x),min=-cl_bound,max=cl_bound) + jnp.clip(jnp.imag(x),min=-cl_bound,max=cl_bound) * 1j
    else:
        return jnp.clip(x,min=-cl_bound,max=cl_bound)

def loss_fn(dect,it,key):
    dec,t = dect
    progress = it / numit
    # temp = 0.0
    # temp = jnp.maximum(0.0,1 - progress)
    temp = jnp.maximum(0.0,1 - 1.1*progress)
    base_loss = basic_loss(dect,temp)
    
    regularization_loss = 0.0
    descretization_loss = 0.0
    for f in dec:
        pass
        
        regularization_loss += 0.02 * jnp.where(progress < 1/3, 0.1 ** (progress*6), 0) * jnp.mean(optax.l2_loss(jnp.abs(f)))
        regularization_loss += 0.1 * jnp.mean(optax.l2_loss(jnp.abs(f-clipped(f))))
        # descretization_loss += ((1-jnp.cos(2*jnp.pi*progress))/2) * 0.02 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f))))
        # descretization_loss += ((1-jnp.cos(2*jnp.pi*progress))/2) * 0.02 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f*2)/2)))
        
        # descretization_loss += 0.003 * jnp.mean(jnp.abs(f))
        # descretization_loss += ((1-jnp.cos(jnp.pi*progress))/2) * 0.03 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f))))
        # descretization_loss += ((1-jnp.cos(jnp.pi*progress))/2) * 0.04 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f*2)/2)))
        # loss += 0.03 * ((1-jnp.cos(jnp.pi*progress))/2) * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f))))
        # loss += 0.01 * ((1-jnp.cos(jnp.pi*progress))/2)**2 * jnp.mean(optax.l2_loss(jnp.abs(f - jnp.round(f*2)/2)))
        # loss += 0.1 * jnp.mean(optax.l2_loss(jnp.abs(f-clipped(f))))
        
    loss = base_loss + regularization_loss + descretization_loss
    return loss,{ 'base loss' : base_loss, 
                    'regularization loss' : regularization_loss, 
                    'descretization loss' : descretization_loss }

start_learning_rate = 0.1
optimizer = optax.adam(start_learning_rate)
# optimizer = optax.adamw(start_learning_rate)
# optimizer = optax.lbfgs(start_learning_rate)
opt_state = jax.vmap(optimizer.init)(dect)

@functools.partial(jax.jit,donate_argnums=[0,1])
@functools.partial(jax.vmap, in_axes=[0,0,0,None])
def update_function(dect, opt_state, key, it):
    (loss, info), grads = jax.value_and_grad(loss_fn,has_aux=True)(dect,it,key)
    grads = jax.tree.map(jnp.conj, grads)
    updates, opt_state = optimizer.update(grads, opt_state, dect, value = loss, grad = grads, value_fn = 
                                          lambda dect: loss_fn(dect,it,key)[0])
    dect = optax.apply_updates(dect, updates)
    return dect, opt_state, loss, info

@jax.jit
def extra_info(opt_state, loss, dect, info):
    lossmult = T.size
    besti = jnp.argpartition(loss,print_num-1)[:print_num]
    besti = besti[jnp.argsort(loss[besti])]
    dect = jax.tree.map(lambda e: e[besti], dect)
    info = jax.tree.map(lambda e: e[besti]*lossmult, info)
    info['example index'] = besti
    print(dect[0][0].shape)
    loss_t0 = jax.vmap(basic_loss,in_axes=[0,None])(dect, 0.0)
    info['base loss temp 0'] = loss_t0 * lossmult
    dec, t = dect
    maxabs = jax.vmap(lambda dec: jax.tree.reduce(jnp.maximum,jax.tree.map(lambda e: jnp.max(jnp.abs(e)), dec)))(dec)
    info['maximum coefficient'] = maxabs
    info['worst batch loss'] = jnp.max(loss)*lossmult
    return info

for it in range(numit):
    if it == 1:
        startt = time.time()
    key, keycur = jax.random.split(key)
    dect, opt_state, loss, info = update_function(dect, opt_state, jax.random.split(keycur,batch), it)
    if it % printevery == printevery-1 or it == numit-1:
        info = extra_info(opt_state,loss,dect,info)
        elapsed = time.time() - startt
        print(f'{it}: {elapsed:.3f}s elapsed, {it*batch / (1000*elapsed):.0f}K iteratons/s')
        for desc, v in sorted(info.items()):
            print(desc.rjust(25), v)
        
# # dec_round = jax.tree.map(lambda x: jnp.round(x),full_dec)
# dec_round = jax.tree.map(lambda x: jnp.round(x*2)/2,full_dec)
# bloss_round = basic_loss_fn(dec_round,keep_pos)
# successes = jnp.count_nonzero(bloss_round == 0)
# sols = jax.tree.map(lambda x: x[bloss_round == 0], dec_round)
# print( f"{successes} solves out of {batch}, {successes/batch*100:.2f}%" )

# # startt = time.time()
# # refine_batch = 1000
# # numrefine = batch
# # numrefine = min(numrefine,batch)
# # loss = basic_loss_fn(dect)
# # besti = jnp.argpartition(loss,numrefine-1)[:numrefine]
# # decbest = jax.tree.map(lambda x: x[besti],dect)
# # @jax.jit
# # # @jax.vmap
# # def refine(dect):
# #     # lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4,verbose=frozenset(['loss']))
# #     lm = optx.LevenbergMarquardt(rtol=1e-3,atol=1e-4,linear_solver=lx.NormalCholesky())
# #     # lm = optx.Newton(rtol=1e-3, atol=1e-4, cauchy_termination=False)
# #     rs_base = residual_function(T)
# #     # # lm = optx.Newton(rtol=1e-3, atol=1e-4, linear_solver=lx.NormalCG(rtol=1e-2,atol=1e-3), cauchy_termination=False)
# #     # lm = optx.Newton(rtol=1e-3, atol=1e-4, linear_solver=lx.AutoLinearSolver(well_posed=True), cauchy_termination=False)
# #     # rs_base = residual_function_fp(T)
# #     def rs_fn(dect, unused):
# #         rs, fdec = rs_base(dect)
# #         rs = [rs]
# #         for f in dect:
# #             rs.append(0.5 * (f - jnp.round(f)))
# #             # rs.append(0.5 * (f - jnp.round(f*2)/2))
# #         #     # rs.append(0.1 * (f - jnp.round(f)))
# #         #     rs.append(0.5 * (f - clipped(f)))
# #         return rs
# #     # return optx.root_find(rs_fn, lm, dect, throw=False).value
# #     return optx.least_squares(rs_fn, lm, dect, throw=False).value
# # # dec_refine = refine(decbest)
# # dec_refine = jax.lax.map(refine, decbest, batch_size=refine_batch)
# # bloss = basic_loss_fn(dec_refine) 
# # besti = jnp.argpartition(bloss,9)[:10]
# # print( bloss[besti] )
# # bloss_round = basic_loss_fn(jax.tree.map(lambda x: jnp.round(x),dec_refine))
# # # bloss_round = basic_loss_fn(jax.tree.map(lambda x: jnp.round(x*2)/2,dec_refine))
# # print( bloss_round[besti] )
# # successes = jnp.count_nonzero(bloss_round == 0)
# # print( f"{successes} solves out of {numrefine}, {successes/numrefine*100:.2f}%" )
# # time_elapsed = time.time() - startt
# # print(f'{time_elapsed:.3f}s elapsed, {numrefine/time_elapsed:.3f} solves/s')
