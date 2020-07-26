import torch
from torch.optim import Optimizer
import math
from torch.autograd import Variable
import numpy as np

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x*y) for (x, y) in zip(xs, ys)])

def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].data.add_(update[i]*alpha) 
    return params

def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v,v)
    s = s ** 0.5
    for i in range(len(v)):
        v[i] = v[i] / (s + 1e-16)
    return v


class STR(Optimizer):
    """Implement Stochastic Trust-Region algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        delta (float, optional): inital trust-region radius (default: 1e-3)
        tol (float, optional): tolerance for residual of sub-problem (default: 1e-6)
        max_iters (int, optional): maximum of inner iterations for sub-problem (default: 25)
        gamma1/2 (float, optional): radius factor for super success/failure (default: 2/1.2)
        rho1/2 (float, optional): super-success/fail threshold for approximation (default: 1.2/0.8)
        max/min_delta (float, optional): the max/min of trust region radius (default: 10/1e-6)
        weight (float, optional): the L2 regularization (default: 0.)
    """

    def __init__(self, params, delta=1, tol=1e-16, max_iters=25, gamma1=2,gamma2=1.2, rho1=0.8, rho2=1e-4, max_delta=10, min_delta=1e-6, weight=0.0):
        defaults = dict(delta=delta, tol=tol, max_iters=max_iters, gamma1=gamma1, gamma2=gamma2, rho1=rho1, rho2=rho2, max_delta=max_delta, min_delta=min_delta, weight=weight)
        super(STR, self).__init__(params, defaults)

    def get_grad(self):
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad)
        return grads

    def get_neg_curvature(self, gradsH, pow_iter=10):
        """
        Implement power iteration for Hessian matrix
        :param gradsH: a list of torch variables
        :return: eig-vector for largest negative eig-value of Hessian, largest negative eig-value, iterations
        """
        tol = 1e-6 
        v = [torch.randn(p.size()).cuda() for p in gradsH]
        v = normalization(v)
        lam_max = 0.
        iter_max = 0

        for i in range(pow_iter):
            #print(i, type(v), type(v[0]))
            #print(i)
            Hv = self.get_hessian(v, gradsH)
            lam_tmp = group_product(Hv, v)
            v = normalization(Hv)
            iter_max += 1
            
            if lam_tmp < 0:
                return v, lam_tmp, iter_max
            if abs(lam_tmp-lam_max) < tol:
                lam_max = lam_tmp
                break
            lam_max = lam_tmp 
    
        u = [torch.randn(p.size()).cuda() for p in gradsH]
        u = normalization(u)        
        lam_min = 0.
        iter_min = 0
        
        for i in range(pow_iter):
            #print(i)
            Hu = self.get_hessian(u, gradsH)
            Hu = [a-lam_max*b for a,b in zip(Hu, u)]
            lam_tmp = group_product(Hu, u)
            u = normalization(Hu)
            iter_min += 1
            if lam_tmp + lam_max < 0:
                return u, lam_tmp + lam_max, iter_max + iter_min
            if abs(lam_tmp-lam_min) < tol:
                lam_min = lam_tmp
                break
            lam_min = lam_tmp

        return u, lam_min+lam_max, iter_max + iter_min


    def get_hessian(self, v, gradsH):
        """
        compute the Hessian vector product with v, at the current gradient point.
        or compute the gradient of <gradsH,v>.
        :param v: a list of torch tensors
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """
                
        params = self.param_groups[0]['params']
        w = self.param_groups[0]['weight']
        hvs = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True)
        hessian_vectors = [hv.data + w*vi for hv, vi in zip(hvs, v)]
        return hessian_vectors     
        

    def get_update(self, gradsH, grads, update=None):
        """
        :param options:
            gradsH: a list of gradients for computing Hessian
            grads: a list of gradients
        :return:
        """
        delta = self.param_groups[0]['delta']
        tol = self.param_groups[0]['tol']
        max_iters = self.param_groups[0]['max_iters']
        weight = self.param_groups[0]['weight']
        grads = [g + weight*p.data for g, p in zip(grads, self.param_groups[0]['params'])]
        p, model, num_iters, flag = self.cg_steihaug(gradsH, grads, delta, tol, max_iters, weight, update)
        return p, model, num_iters, flag
        
    def cg_steihaug(self, grads_fun, grads_data, delta, tol, max_iters, weight=0.0, update=None):
        """
        CG-Steighaug algorithms for approximately solving
            min Q(p) = 1/2 p' H p + g'p s.t. ||p|| <= delta
        assume the parameters are in groups.

        :param grads_fun: a list of gradients for computing Hessian
        :param grads_data: a list of gradients
        :param delta: trust-region radius
        :param tol: relative residual
        :param max_iters: maximum iterations

        :return: p: an approximate solution, a list of tensors
                 num_cg: number of CG iterations
                 flag: termination condition
        """
        
        gnorms = group_product(grads_data, grads_data)
        params = self.param_groups[0]['params']
        zs = [0.0*p.data for p in params]
        rs = [g.data + weight*p.data for g,p in zip(grads_data, params)]
        ds = [-g.data - weight*p.data for g,p in zip(grads_data, params)]
        
        for i in range(max_iters):
            Hd = torch.autograd.grad(grads_fun, params, grad_outputs=ds, only_inputs=True, retain_graph=True)
            Hd = [hd.data + weight*d for hd, d in zip(Hd, ds)]
            dHd = group_product(Hd, ds)
            if dHd <= 0.0:
                ac = group_product(ds, ds)
                bc = 2 * group_product(zs, ds)
                cc = group_product(zs, zs) - delta*delta
                tau = (-bc + math.sqrt(bc*bc - 4*ac*cc))/(2*ac+1e-6)
                flag = "Negative Curvature"
                ps = [z + tau*d+0.0 for (z, d) in zip(zs, ds)]
                break

            rnorm_square = group_product(rs, rs)
            alpha = rnorm_square/(dHd + 1e-6)
            zs_next = [z + alpha*d+0.0 for z,d in zip(zs, ds)]
            zs_next_norm = math.sqrt(group_product(zs_next, zs_next))
            if zs_next_norm >= delta:
                ac = group_product(ds, ds)
                bc = 2 * group_product(zs, ds)
                cc = group_product(zs, zs) - delta*delta
                tau = (-bc + math.sqrt(bc*bc - 4*ac*cc))/(2*ac+1e-6)
                flag = "Hit Boundary"
                ps = [z + tau*d+0.0 for (z, d) in zip(zs, ds)]
                break
            zs = [z+0.0 for z in zs_next]
            rs = [r + alpha*hd+0.0 for r, hd in zip(rs, Hd)]
            rnext_norm_square = group_product(rs, rs)
            if rnext_norm_square < tol:
                flag = "Small Residue"
                ps = [z+ 0.0 for z in zs]
                break
            beta = rnext_norm_square/(rnorm_square+1e-6)
            ds = [-r + beta*d+0.0 for r, d in zip(rs, ds)]
        if i == max_iters - 1:
            flag = "Maximum Iterations"
            ps = [z+0.0 for z in zs]
        Hp = torch.autograd.grad(grads_fun, params, grad_outputs=ps, only_inputs=True, retain_graph=True)
        Hp = [hp.data+0.0 for hp in Hp]
        m_obj = 0.5*group_product(Hp, ps) + group_product(ps, grads_data)+ 0.5*weight*group_product(ps,ps)
        return ps, m_obj.item(), i+1, flag
        
        

        
    def step(self, gradsH, grads, closure=None,loss=None):
        """
        Perform a TR step
        :param grads_fun: a list of gradients for computing Hessian
        :param grads_data: a list of gradients
        :param closure:
        :return:
        """
        if closure == None:
            update, model, num_iters, flag = self.get_update(gradsH, grads)
            group_add(self.param_groups[0]['params'], update)
            return None, model, num_iters, 0
        if loss==None:
            cur_loss = closure()
        else:
            cur_loss = loss
        inner_iters = 0

        # for positive check
        num_loss = 0
        update = None
        while 1:
            update, model, num_iters, flag = self.get_update(gradsH, grads)
            inner_iters += num_iters
            # print ("num_iters", num_iters, flag)
            # print ("delta", self.param_groups[0]['delta'])
            
            group_add(self.param_groups[0]['params'], update);
            loss = closure()
            decr = loss - cur_loss
            appr = decr/(model - 1e-16)
            # print (loss, cur_loss, model, appr)
            if model >= 0.0:
                # print('model positive\n')
                if decr < 0:
                    break
                else:
                    group_add(self.param_groups[0]['params'], update, -1)
                    break
                    
            if appr > self.param_groups[0]['rho1']:
                # super success
                self.param_groups[0]['delta'] = min(self.param_groups[0]['delta']*self.param_groups[0]['gamma1'], self.param_groups[0]['max_delta'])
                break
            elif appr > self.param_groups[0]['rho2']:
                # success
                self.param_groups[0]['delta'] = min(self.param_groups[0]['delta']*self.param_groups[0]['gamma2'], self.param_groups[0]['max_delta'])
                break
            else:
                # failure
                self.param_groups[0]['delta'] = max(self.param_groups[0]['delta']/self.param_groups[0]['gamma1'], self.param_groups[0]['min_delta'])
                # break # even if the method is fail, still update it, like what sgd does
                group_add(self.param_groups[0]['params'], update, -1)
                
            num_loss += 1
            
            # if minibatch of data set is not good, break it and do another sampling. 
            if num_loss == 3:
                break

        
        return loss, model, inner_iters, num_loss


  
