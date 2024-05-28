#!/usr/bin/env python

import numpy as np
import scipy.stats
import pandas
import pickle
import os.path
import sys

def _logit(x):
    return np.log(x) - np.log(1 - x)

def _logistic(x):
    return 1/(1 + np.exp(-1*x))

def _rand_prob_vec(n, prng):
    a = prng.random(n) + 0.001
    return a/sum(a)
    
def _norm_logp(p):
    b = np.max(p, axis=1)[:, None]
    cll_norm = [np.exp(p[i0] - b[i0] - np.log(np.sum(np.exp(p[i0] - b[i0])))) for i0 in range(len(p))]
    return np.array(cll_norm)

def _logsumexp(p):
    b = np.max(p, axis=1)[:, None]
    lse = [b[i0] + np.log(np.sum(np.exp(p[i0] - b[i0]))) for i0 in range(len(p))]
    return lse

def _init_dist_params(dist_type, minv, maxv, M, data, prng):
    p = prng.random()/2
    if dist_type == 'C':
        k = data.unique()
        n = len(k)
        pv = _rand_prob_vec(n, prng)
        pv = pv - 0.1*np.min(pv)
        pd = {k[i0]: pv[i0] for i0 in range(n)}
        return (n, [p, pd])
    elif dist_type == 'O':
        return (3, [p, [prng.random()*(maxv - minv) + minv, ((maxv - minv)/M/3)**2]])
    elif dist_type == 'R':
        m0 = data.min()
        m1 = data.max()
        return (3, [p, [prng.random()*(m1 - m0) + m0, ((m1 - m0)/M)**2]])
    elif dist_type == 'PR':
        m0 = data.min()
        m1 = data.max()
        a = prng.random()*(m1 - m0) + m0
        b = ((m1 - m0)/M)**2
        pz = prng.random()/4
        return (4, [p, [pz, b/a, a**2/b]])
    elif dist_type == 'I':
        data1 = data[data != 1]
        m0 = _logit(data1.min())
        m1 = _logit(data1.max())
        po = prng.random()/6
        sm = 0.001
        return (4, [p, [po, prng.random()*(m1 - m0) + m0, sm + ((m1 - m0)/M)**2]])

def _init_params(model_def, M, data, prng):
    var_names = list(model_def.keys())

    params = [_rand_prob_vec(M, prng)]
    num_params = M - 1
    for m0 in range(M):
        params_m = {}
        for v0 in var_names:
            v_inf = model_def[v0]
            (np0, params_m[v0]) = _init_dist_params(v_inf[0], v_inf[1], v_inf[2], M, data[v0], prng)
            num_params = num_params + np0
        params.append(params_m)

    return (num_params, params)

def _quant_g(minv, maxv, m, v):
    pv = [scipy.stats.norm(m, np.sqrt(v)).pdf(i0) for i0 in range(minv, maxv + 1)]
    pv = pv/sum(pv)

    return pv

def _sample(model_def, params, prng):
    var_names = list(model_def.keys())
    a = params[0]
    M = len(a)

    m = prng.choice(range(M), p=a)
    params_m = params[m + 1]
    s = {}
    for v0 in var_names:
        vt = model_def[v0][0]
        params_mv = params_m[v0]
        missp = params_mv[0]
        varp = params_mv[1]
        missing = prng.choice([True, False], p=[missp, 1 - missp])
        if missing:
            s[v0] = None
        else:
            if vt == 'C':
                pv = varp.values()
                pv = np.array(list(pv))/sum(pv)
                s[v0] = prng.choice(list(varp.keys()), p=pv)
            elif vt == 'O':
                minv = model_def[v0][1]
                maxv = model_def[v0][2]
                pv = _quant_g(minv, maxv, varp[0], varp[1])
                s[v0] = int(prng.choice(range(minv, maxv + 1), p=pv))
            elif vt == 'R':
                s[v0] = prng.normal(varp[0], np.sqrt(varp[1]))
            elif vt == 'PR':
                z = prng.choice([True, False], p=[varp[0], 1 - varp[0]])
                if z:
                    s[v0] = 0
                else:
                    s[v0] = prng.gamma(varp[1], varp[2])
            elif vt == 'I':
                z = prng.choice([1, 0.5], p=[varp[0], 1 - varp[0]])
                if z == 1:
                    s[v0] = 1
                else:
                    s[v0] = _logistic(prng.normal(varp[1], np.sqrt(varp[2])))

    return s

def _loglike(dist_type, minv, maxv, data, params):
    missp = params[0]
    varp = params[1]
    N = len(data)
    ll = np.empty(N)
    ll[:] = np.nan
    missing_idx = pandas.isnull(data)
    ll[missing_idx] = np.log(missp)
    N_nm = sum(~missing_idx)
    if dist_type == 'C':
        k = 1 - sum(varp.values())
        ll[~missing_idx] = [np.log(1 - missp) + np.log(varp.get(d0, k)) for d0 in data[~missing_idx]]
    elif dist_type == 'O':
        pv = scipy.stats.norm(varp[0], np.sqrt(varp[1])).pdf(range(minv, maxv + 1))
        if np.sum(pv) == 0:
            pv[:] = 1./len(pv)
        else:
            pv = pv/np.sum(pv)
        pv = pv*N_nm + 1
        pv = pv/np.sum(pv)

        ll[~missing_idx] = [np.log(1 - missp) + np.log(pv[d0 - minv]) for d0 in data[~missing_idx]]
    elif dist_type == 'R':
        ll[~missing_idx] = np.log(1 - missp) + scipy.stats.norm(varp[0], np.sqrt(varp[1])).logpdf(data[~missing_idx])
    elif dist_type == 'PR':
        z_idx = np.where(data == 0)
        ll[z_idx] = np.log(1 - missp) + np.log(varp[0])
        pos_idx = np.where(data > 0)[0]
        ll[pos_idx] = np.log(1 - missp) + np.log(1 - varp[0]) + scipy.stats.gamma(varp[1], loc=0, scale=varp[2]).logpdf(data[pos_idx])
    elif dist_type == 'I':
        o_idx = np.where(data == 1)
        ll[o_idx] = np.log(1 - missp) + np.log(varp[0])
        idx = np.where(data < 1)[0]
        data1 = data[idx].map(_logit)
        ll[idx] = np.log(1 - missp) + np.log(1 - varp[0]) + scipy.stats.norm(varp[1], np.sqrt(varp[2])).logpdf(data1)

    return ll

def _probs(model_def, params, data):
    var_names = list(model_def.keys())

    a = params[0]
    M = len(a)
    p = []
    for m0 in range(M):
        pv = []
        for v0 in var_names:
            v_inf = model_def[v0]
            v_params = params[m0 + 1][v0]
            pv.append(_loglike(v_inf[0], v_inf[1], v_inf[2], data[v0], v_params))

        p.append(pv)

    return np.array(p).transpose([2, 0, 1])
 
def _est_dist_params(dist_type, data, w):
    N = len(data)
    missing_idx = pandas.isnull(data)
    missing_p = (np.sum(w[missing_idx]) + 1)/(sum(w) + 1)
    data_filled = data.dropna()
    w_filled = w[~missing_idx]
    if dist_type == 'C':
        k = data_filled.unique()
        p_unob = 1/(2 + sum(w_filled))
        f_unob = p_unob/len(k)
        x1 = {k0: (sum(w_filled[data_filled == k0]) + 1/len(k))/(sum(w_filled) + 1) - f_unob for k0 in k}
        x2 = sum(x1.values())
        pd = {i0:x1[i0]/x2 for i0 in x1}
        return [missing_p, pd]
    elif dist_type == 'O' or dist_type == 'R':
        if sum(w_filled) == 0:
            print('A')
        mu = np.dot(w_filled, data_filled)/sum(w_filled)
        sm = (np.std(data)/100)**2
        v = sm + np.dot(w_filled, (data_filled - mu)**2)/sum(w_filled)
        if v == 0:
            print('B')
        return [missing_p, [mu, v]]
    elif dist_type == 'PR':
        z_idx = np.where(data == 0)
        p_z = (np.sum(w[z_idx]) + 1)/(sum(w) + 1)
        pos_idx = np.where(data > 0)[0]
        s = np.log(np.dot(w[pos_idx], data[pos_idx])/sum(w[pos_idx])) - np.dot(w[pos_idx], np.log(data[pos_idx]))/sum(w[pos_idx])
        if s == 0:
            print('C')
            mu = np.dot(w[pos_idx], data[pos_idx])/sum(w[pos_idx])
            sm = (np.std(data)/100)**2
            k = mu**2/sm
            theta = sm/mu
            return [missing_p, [p_z, k, theta]]
        k = (3 - s + np.sqrt((s - 3)**2 + 14*s))/(12*s)
        theta = np.dot(w[pos_idx], data[pos_idx])/sum(w[pos_idx])/k
        if k == 0:
            print('D')
        if theta == 0:
            print('E')
        return [missing_p, [p_z, k, theta]]
    elif dist_type == 'I':
        o_idx = np.where(data == 1)
        idx = np.where(data < 1)[0]

        p_o = (np.sum(w[o_idx]) + 1)/(sum(w) + 1)
        p_1 = (np.sum(w[idx]) + 1)/(sum(w) + 1)
        s = p_o + p_1
        p_o = p_o/s

        if sum(w[idx]) == 0:
            print('A-I')

        data1 = data[idx].map(_logit)
        mu = np.dot(w[idx], data1)/sum(w[idx])
        sm = 0.001
        v = sm + np.dot(w[idx], (data1 - mu)**2)/sum(w[idx])
        return [missing_p, [p_o, mu, v]]


def _est_params(model_def, M, data, w):
    var_names = list(model_def.keys())
    M = w.shape[1]

    w = w + 0.001
    w = w/np.sum(w, axis=1)[:, None]
    params = [np.mean(w, axis=0)]
    for m0 in range(M):
        params_m = {}
        for v0 in var_names:
            v_inf = model_def[v0]
            params_m[v0] = _est_dist_params(v_inf[0], data[v0], w[:, m0])
        params.append(params_m)

    return params


def _write_chkpt(i0, ll, cr, params, chkpnt):
    with open(chkpnt + '.iter', 'a') as fid:
        if ll is None:
            fid.write('%i, %f'%(i0, cr))
        elif cr is None:
            fid.write(', %f\n'%ll)
        else:
            fid.write(', %f\n'%ll)
            fid.write('%i, %f'%(i0, cr))

    with open(chkpnt + '.params', 'wb') as fid:
        pickle.dump(params, fid)

def _load_chkpnt(chkpnt):
    iter_num = []
    ll = []
    cr = []
    with open(chkpnt + '.iter', 'r') as fid:
        for line in fid:
            x = line.split(',')
            iter_num.append(int(x[0].strip()))
            cr.append(float(x[1].strip()))
            if len(x) == 3:
                ll.append(float(x[2].strip()))

    with open(chkpnt + '.params', 'rb') as fid:
        params = pickle.load(fid)

    return (iter_num, ll, cr, params)


class hdtmm:
    def __init__(self, model_def, M):
        # model_def is a dictionary
        #   var_name: [type, min, max, direction]
        #
        #      type: one of 'C', 'O', 'R', 'PR', 'I'
        #      min: minimum value or None
        #      max: maximum value of None
        #      direction: True for bad --> good
        #                 False for good --> bad
        #                 None if not applicable
        #
        #   M is the number of latent states
        #

        self.model_def = model_def
        self.M = M

    def init_params(self, data, randstate=None):
        if randstate is not None:
            prng = np.random.RandomState(randstate)
        else:
            prng = np.random.RandomState()
        (num_params, p) = _init_params(self.model_def, self.M, data, prng)
        self.num_params = num_params
        self.params = p

    def sample(self, N, randstate=None):
        if randstate is not None:
            prng = np.random.RandomState(randstate)
        else:
            prng = np.random.RandomState()
        data = []
        for n0 in range(N):
            data.append(_sample(self.model_def, self.params, prng))

        dtypes_Int64 = {i0: 'Int64' for i0 in self.model_def.keys() if self.model_def[i0][0] == 'O'}
        return pandas.DataFrame(data).astype(dtypes_Int64)

    def Estep(self, data, P=None):
        if P is None:
            P = _probs(self.model_def, self.params, data)
        P = np.sum(P, axis=2)
        P = P + np.log(self.params[0])
        P_norm = _norm_logp(P)

        return P_norm

    def Mstep(self, data, w):
        self.params = _est_params(self.model_def, self.M, data, w)

    def LL(self, data, P=None, persample=False):
        if P is None:
            P = _probs(self.model_def, self.params, data)  # N x M x V
        P = np.sum(P, axis=2)
        P = P + np.log(self.params[0])
        Plse = np.array(_logsumexp(P))
        if persample:
            return Plse
        else:
            return np.sum(Plse)

    def estimate(self, data, verbose=False, th=0.001, chkpnt=None, randstate=None):
        # data is a pandas dataframe
        #    variable names must match var_names in self.model_def
        #

        N = len(data)
        ll_n = N*len(self.model_def)
        
        if chkpnt is not None:
            if os.path.exists(chkpnt + '.iter'):
                (it_nums, lls, crs, params) = _load_chkpnt(chkpnt)
                self.__dict__ = params
                p_old = self.params[0]
                it_c = it_nums[-1]
                for i0 in range(it_c):
                    sys.stdout.write('Iteration %i: %f, %f\n'%(it_nums[i0], lls[i0]/ll_n, crs[i0]))
                    sys.stdout.flush()
                i0 = it_nums[-1]
                ll_best = max(lls)
                cr = crs[-1]
                if len(lls) == len(crs):
                    sys.stdout.write('Iteration %i: %f, %f\n'%(it_nums[-1], lls[-1]/ll_n, crs[-1]))
                    sys.stdout.write('Final model: %f, BIC %f\n'%(self.ll/ll_n, self.bic/ll_n))
                    sys.stdout.flush()
                    return
            else:
                i0 = 0
        else:
            i0 = 0
        
        if i0 == 0:
            self.init_params(data, randstate=randstate)
            p_old = self.params[0]
            cr = np.inf
            ll_best = -np.inf
            if chkpnt is not None:
                _write_chkpt(0, None, cr, self.__dict__, chkpnt)

        while cr > th:
            P = _probs(self.model_def, self.params, data)
            ll = self.LL(data, P)
            if ll > ll_best:
                ll_best = ll
                self.params_best = self.params
            else:
                print('* ' + str(ll_best/ll_n))
            if verbose:
                sys.stdout.write('Iteration %i: %f, %f\n'%(i0, ll/ll_n, cr))
                sys.stdout.flush()
            i0 = i0 + 1

            a = self.Estep(data, P)
            self.Mstep(data, a)
            p_new = self.params[0]
            cr = np.max(np.abs(p_new - p_old))
            p_old = p_new
            if chkpnt is not None:
                _write_chkpt(i0, ll, cr, self.__dict__, chkpnt)

        P = _probs(self.model_def, self.params, data)
        ll = self.LL(data, P)
        if ll > ll_best:
            ll_best = ll
            self.params_best = self.params
        if verbose:
            sys.stdout.write('Iteration %i: %f, %f\n'%(i0, ll/ll_n, cr))
            sys.stdout.flush()
        self.ll = ll_best
        self.bic = self.num_params*np.log(N) - 2*ll_best
        self.params = self.params_best
        if chkpnt is not None:
            _write_chkpt(0, ll, None, self.__dict__, chkpnt)

        sys.stdout.write('Final model: %f, BIC %f\n'%(ll_best/ll_n, self.bic/ll_n))
