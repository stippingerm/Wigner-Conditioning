# -*- coding: utf-8 -*-
"""
Created on Jul 27, 2016
Last major modification Jul 29, 2016
Last major bug fix on Aug 01, 2016
Last major revision Aug 29, 2016

# A Non-Parametric Bayesian Method for Inferring Hidden Causes
F., Griffiths, T.L., Ghahramani, Z., 2006.
Proceedings of the Conference on Uncertainty in Artificial Intelligence.
See: http://cocosci.berkeley.edu/tom/papers/ibpuai.pdf

@author: Marcell Stippinger
"""

import numpy as np
import scipy.special, scipy.stats
import warnings
gamma = scipy.special.gamma
gammaln = scipy.special.gammaln
poisson = scipy.stats.poisson
bernoulli = scipy.stats.bernoulli
beta = scipy.stats.beta


### General functions for pprobability manipulation
def p0_per_sum_p(p=None, logp=None):
    '''Get the relative weight of the first element in each column
       (i.e. along axis 0)'''
    # Return value is alwys given in linear form
    assert (p is None) != (logp is None)
    if logp is None:
        p = np.array(p, ndmin=1)
        assert np.all(p>=0)
        ret = p[0] / np.sum(p, axis=0)
        return ret
    else:
        logp = np.array(logp, ndmin=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            frac = np.exp(logp - logp[0])
            ret = 1.0 / np.sum(frac, axis=0)
        if logp.ndim > 1:
            ret[logp[0] == -np.inf] = 0
        elif logp[0] == -np.inf:
            ret = 0
        return ret

def normalize_p(p=None, logp=None, axis=None):
    '''Normailze weights along a given axis, default is None (whole array)'''
    assert (p is None) != (logp is None)
    if logp is None:
        p = np.array(p, ndmin=1)
        assert np.all(p>=0)
        ret = p / np.sum(p, axis=axis)
    else:
        import warnings
        logp = np.array(logp, ndmin=1)
        if axis is None:
            flat = logp.ravel()
        else:
            flat = np.swapaxes(np.expand_dims(logp, axis=-1), axis, -1)
        #assert logp.ndim == 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            frac = np.exp(flat - np.expand_dims(logp, axis=-1))
            # nans come from (-inf-(-inf)), we could safely set to inf
            # since -inf corresponds to 0 probability
            #frac[np.isnan(frac)] = np.inf
            ret = 1.0 / np.sum(frac,axis=-1)
        ret[logp == -np.inf] = 0
    if np.any(np.all(ret==0, axis=axis)):
        import warnings
        print('p',p,'logp',logp)
        warnings.warn(
            'The provided values cannot be normalized along axis=%s'%axis)
    return ret


### Data manipulation
def MakeArray(data):
    '''Make numpy array out of data'''
    if type(data) is np.ndarray:
        return data
    try:
        data = np.array(data, ndmin=1)
        return data
    except:
        print ('Data could not be converted to numpy.ndarray')
        raise


def plot_matrix_product(rows, M1, name1, cols, M2, name2,
                        common, result=None, namer='result', figsize=(20,5)):
    '''Visualize a matrix product'''
    # names of axes
    assert type(rows) is str and type(cols) is str and type(common) is str
    # names of matrices
    assert type(name1) is str and type(name2) is str and type(namer) is str
    # matrices
    assert type(M1) is np.ndarray and type(M2) is np.ndarray and (
                        result is None or type(result) is np.ndarray)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2,sharex='col',sharey='row',figsize=figsize)
    ax[0,0].axis('off')
    ax[0,1].matshow(M2)
    ax[0,1].set_title(name2)
    ax[0,1].set_ylabel(common)
    ax[1,0].matshow(M1)
    ax[1,0].set_title(name1)
    ax[1,0].set_xlabel(common)
    ax[1,1].matshow(np.dot(M1,M2) if result is None else result)
    ax[1,1].set_title(namer)
    ax[1,1].set_xlabel(cols)
    ax[1,1].set_ylabel(rows)
    return fig


### amortization accounter

class Amortizer(object):
    '''Easy-to-use interface to amortization through decay_time'''
    def __init__(self, decay_time = np.inf):
        '''Initialize with given decay time or no amortization'''
        assert decay_time > 0
        self.decay_time = decay_time
        self.clear()
        
    def clear(self):
        '''Clear all'''
        self.loga = np.ndarray((0,))
        
    def calc_loga(self, n_samples, amortization=None):
        '''Logarithmic amortization increments'''
        if amortization is None:
            loga = -np.log(2.0) / self.decay_time * np.ones((n_samples,))
        else:
            assert np.all((0<amortization)&(0<amortization<=1))
            loga = np.broadcast_to(np.log(amortization), (n_samples,))        
        return loga
    
    @staticmethod
    def calc_logw(loga):
        # cumsum from the end but return in original order
        return np.cumsum(loga[::-1])[::-1]

    def __getattr__(self, name):
        '''Provide calculated values'''
        if name in ['logw']:
            # cumsum from the end but return in original order
            return self.calc_logw(self.loga)
        if name in ['w']:
            return np.exp(self.logw)
        #raise AttributeError

    def __setattr__(self, name, value):
        '''Protect calculated values'''
        if name in ['logw', 'w']:
            raise AttributeError('Cannot set calulated value')
        super(Amortizer,self).__setattr__(name,value)

### Bayesian inference

class BayesianNetwork(object):
    '''Bayesian network interface:
       X: observations, shape=[n_features,n_trials]
       Y: causes, shape=[n_causes,n_trials]
       A: adjacency, shape=[n_features,n_features]
       Z: causality, shape=[n_features,n_causes]
       N==n_features, T==n_samples==n_trials, K==n_causes
       i€(1,...,N), t€(1,...,T), k€(1,...,K)'''

    def observe(self, X, amortization=None):
        '''Cumulate observations, n_features must be the same as before'''
        X = np.array(X, ndmin=1)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)
        elif X.ndim > 2:
            raise ValueError('X must be 1 or 2 dimensional')

        if len(self.X):
            self.X = np.row_stack((self.X,X))
        else:
            self.X = X

        new_samples = X.shape[1]
        self.px = np.zeros_like(X)
        
        loga = self.amo.calc_loga(new_samples, amortization)
        self.amo.loga = np.append(self.loga,loga)
        #n_step = self.T - 1.0
        #start = -np.log(2) * (n_step/decay_time)
        #self.logw = np.linspace(start,0.0,self.T)

    def clear(self):
        '''Clear all observations'''
        self.X = np.ndarray((0,0))
        self.Y = np.ndarray((0,0))
        self.px = np.ndarray((0,0))
        self.amo.clear()

    def reset(self):
        '''Reset inner state'''
        self.clear()
        self.A = np.ndarray((0,0))
        self.Z = np.ndarray((0,0))

    def __init__(self, decay_time = np.inf):
        '''Initialize with empty arrays'''
        self.amo = Amortizer(decay_time)
        self.reset()

    def __getattr__(self, name):
        '''Provide calculated values'''
        if name in ['n_features', 'N']:
            return self.X.shape[0]
        if name in ['n_samples', 'n_trials', 'T']:
            return self.X.shape[1]
        if name in ['n_causes', 'K']:
            return self.Y.shape[0]
        if name in ['loga']:
            return self.amo.loga
        if name in ['logw']:
            return np.cumsum(self.loga[::-1])[::-1]
        if name in ['w']:
            return np.exp(self.logw)
        #raise AttributeError

    def __setattr__(self, name, value):
        '''Protect calculated values'''
        if name in ['n_features', 'N',
                     'n_samples', 'n_trials', 'T',
                     'n_causes', 'K',
                     'loga', 'logw', 'w']:
            raise AttributeError('Cannot set calulated value')
        super(BayesianNetwork,self).__setattr__(name,value)

    def get_p(self):
        '''Measure activation'''
        return float(np.sum(self.Y))/np.prod(self.Y.shape)

def binary_iterator(size):
    '''Iterate over binary matrices of given size'''
    import itertools
    size = np.array(size).astype(int)
    length = int(np.prod(size))
    assert (length > 0)
    # Possible choices for elements of s
    a=([0,1],)*length
    for s in itertools.product(*a):
        yield np.array(s).reshape(size)

#def increasing_series(mylist, level, strict=False):
#    mylist = list(mylist)
#    level = int(level)
#    assert level>0
#    if level == 1:
#        for myval in mylist:
#            yield (myval, )
#    else:
#        for pos, myval in enumerate(mylist):
#            otherlist = mylist[(pos+1 if strict else pos):]
#            for otherval in increasing_series(otherlist, level-1, strict):
#                yield (myval,) + otherval

def increasing_series(mylist, level, strict=False, start=0):
    level = int(level)
    assert level>0
    if level == 1:
        for pos in range(start, len(mylist)):
            myval = mylist[pos]
            yield (myval, )
    else:
        for pos in range(start, len(mylist)):
            myval = mylist[pos]
            otherstart = (pos+1 if strict else pos)
            for otherval in increasing_series(mylist, level-1, strict, otherstart):
                yield (myval,) + otherval

def lof(Z):
    '''Left-ordered form'''
    # move first row to the last, second to the second last, etc.
    # because default ordering is according to last row but we want acc. to first
    Q = np.flipud(Z)
    # sort
    s = np.lexsort(Q)
    Q = Z[:,s]
    # move first column to the last, second to the second last, etc.
    # because default ordering is increasing but we want decreasing order
    Q = np.fliplr(Q)
    return Q

#def lof_iterator(size):
#    '''Iterate over binary matrices of left-ordered form of given size'''
#    # Not optimized
#    import itertools
#    return itertools.ifilter(lambda m: np.all(m==lof(m)), binary_iterator(size))

def lof_iterator(size):
    '''Iterate over binary matrices of left-ordered form of given size'''
    N, K = size
    # Change MSB either here or by fliplr later
    # columns = list(reversed(list(binary_iterator(N))))
    columns = list(binary_iterator(N))
    for ibp in increasing_series(columns, K):
        yield np.fliplr(np.column_stack(ibp))


def remove_empty(Z, axis=0):
    '''Remove columns with zeros from a 2d matrix'''
    if Z.ndim != 2:
        raise ValueError('The array must be 2d')
    if axis < 0:
        raise ValueError('Only positive axis id is accepted')
    keep = np.any(Z, axis=axis)
    # take, select, choose work differently, fancy indexing tries flattened...
    ret = np.compress(keep, Z, axis=1-axis)
    return ret

# Model parameters: alpha, p, lambda, epsilon
# Priors on: K (number of causes), Y (activation), Z (network)
# Observation: X
class BernoulliBetaAssumption(BayesianNetwork):
    '''Bayesian inference assuming that all causes are active independently
       with probability p and they are related to observables via a noisy-OR
       distribution according to the network of relations: lambda describes
       effectiveness and eps the baseline probability.
       The network has edges for cause k with Bernoulli(theta[k]) where
       theta[k] are independent from Beta(alpha/K, 1).
       Call Gibbs_prepare first, then Gibbs_iterate as many times as needed'''

    def __init__(self, p, alpha, lamb=0.9, eps=0.01, **kwarg):
        '''Initialize model parameters and check their domain'''
        alpha, p = float(alpha), float(p)
        lamb, eps = float(lamb), float(eps)
        assert (0<alpha)
        assert (0<p) and (p<1)
        assert (0<lamb) and (lamb<=1)
        assert (0<=eps) and (eps<1)
        self.alpha, self.p = alpha, p
        self.lamb, self.eps = lamb, eps
        super(BernoulliBetaAssumption, self).__init__(**kwarg)

    def __gen_Y(self, K, T):
        Y = bernoulli.rvs(self.p, size=(K, T))
        return Y

    def __gen_Z(self, N, K):
        theta = beta.rvs(a=self.alpha/K, b=1, size=K)
        # Buggy: Z = bernoulli.rvs(theta, size=(N, K))
        Z = bernoulli.rvs(np.broadcast_to(theta,(N, K)))
        success = np.all(self.m(Z))
        return success, Z

    def __gen_data(self, Y, Z):
        X = bernoulli.rvs(self.P_x_YZ(Y, Z))
        return X

    def IBP(self, N=None, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if N is None:
            N = self.N
        alpha, N = float(alpha), int(N)
        K_new = poisson.rvs(alpha/np.arange(1,N+1))
        K_new = np.hstack(([0], np.cumsum(K_new)))
        K = K_new[-1]
        m = np.zeros((K,)).astype(float)
        Z = np.zeros((N,K))
        for i in range(0,N):
            Z[i,:] = bernoulli.rvs(m/(i+1))
            Z[i,K_new[i]:K_new[i+1]] = 1
            m += Z[i]
        return Z

    def generate(self, K, N, T, data=False, retries=1000):
        '''Generate a model and corresponding data where the Bayesian network is
           established via rejection-sampling from the Indian Buffet Process.'''
        N, K, T = int(N), int(K), int(T)
        assert (N>0) and (K>=0) and (T>0)
        # causes active
        Y = self.__gen_Y(K, T)
        # rejection-sample the network
        while True:
            success, Z = self.__gen_Z(N, K)
            if success:
                break
            retries -= 1
            if not retries:
                raise ValueError('Could not properly initialize Z')

        # generate observations if requested
        if data:
            X = self.__gen_data(Y, Z)
            return X, Y, Z
        else:
            return Y, Z

    # Initialization for Algo 2
    def Gibbs_prepare(self, K=0, *arg, **kwarg):
        '''Initialize Gibbs-sampler in a random state with K causes.
           The number of causes to be learned is potentially unlimited.'''
        assert type(K) is int
        assert 0 <= K
        self.Y, self.Z = self.generate(K, self.N, self.T, False, *arg, **kwarg)
        self.K_effective = np.inf

    # Initialization for Algo 2
    def Gibbs_load(self, Y, Z):
        '''Initialize Gibbs-sampler in a saved state.
           The number of causes to be learned is potentially unlimited.'''
        K1, T = Y.shape
        N, K2 = Z.shape
        assert (K1 == K2) and (N == self.N) and (T == self.T)
        assert np.all((Y==0)|(Y==1)) and np.all((Z==0)|(Z==1))
        self.Y, self.Z = Y, Z
        self.K_effective = np.inf

    # Eq. 3 used to generate observations
    def P_x_YZ(self, Y=None, Z=None):
        '''Likelihood of observing feature i in trial t: P(x[i,t]==1|Y,Z) from
           the noisy-OR distribution according to the model and the latent
           activity provided. Use this function to replace previous self.Px()'''
        # variables
        if Y is None:
            Y = self.Y
        if Z is None:
            Z = self.Z
        # Z[i,k]@Y[k,t] -> (i,t)
        pure_scalar_prod = np.dot(Z[:,:],Y[:,:])
        # ... -> (i,t)
        prob = 1 - np.power(1-self.lamb,pure_scalar_prod) * (1-self.eps)
        return prob

    # Eq. 5 prior of finite model
    def logP_Z(self, Z):
        '''Log likelihood of a binary matrix Z among matrices of the same size'''
        N, K = Z.shape
        m = np.sum(Z, axis=0)
        a = self.alpha
        # probability atom
        log_p_k = np.log(a/K) + gammaln(m+a/K) + gammaln(N-m+1) - gammaln(N+1+a/K)
        return np.sum(log_p_k)

    # Eq. 5b prior of infinite model
    def logP_Z_inf(self, Z):
        '''Log likelihood of a binary matrix Z among matrices of unlimited number
           of columns (causes)'''
        import itertools
        Z = lof(Z)
        Z = remove_empty(Z)
        N, K = Z.shape
        m = self.m(Z, keepdims=False)
        a = self.alpha
        # probability atom
        log_p_k = gammaln(N-m+1) + gammaln(m) - gammaln(N+1)
        # count the occurrences (len) of columns (c)
        K_h = [len(list(g)) for k, g in itertools.groupby(tuple(c) for c in Z.T)]
        K_h = np.array(K_h, ndmin=1)
        # harmonic sum
        H_N = np.sum(1.0/np.arange(1,N+1))
        # the prefactor
        log_prefactor = K * np.log(a) - np.sum(gammaln(K_h+1)) + a * H_N
        # the product
        ret = log_prefactor + np.sum(log_p_k)
        return ret

    def logP_Z_X(self, Z=None, X=None, w=None):
        '''Enumarate log posterior of the network Z over all possible latent
           constellations Y'''
        import itertools
        
        X_self = X is None
        if X is None:
            X = self.X
        N, T = X.shape
        if Z is None:
            Z = self.Z
        N_, K = Z.shape
        assert N == N_

        if w is None:
            if X_self:
                w = self.w
            else:
                w = self.amo.calc_logw(self.amo.calc_loga(X.shape[1]))
        w = np.array(w, ndmin=1)
        assert (w.ndim == 1) and ((len(w)==1) or len(w)==T)
        
        # "Cumulative" storing sum_Y1 ... -> (t)
        cum = np.zeros((T))
        # Prior for choosing Z -> ()
        prob0 = self.logP_Z_inf(Z)
        # Possible choices for elements of Y1
        a=([0,1],)*K
        # Y1[k] -> (k)
        for Y1 in itertools.product(*a):
            Y1 = np.array(Y1, ndmin=1)
            # Y[k,t] with identic columns -> (k,t)
            # therefore we are not allowed to eliminate t within the loop
            Y = np.broadcast_to(Y1[:,np.newaxis], (K,T))
            # Prior for choosing Y1 -> ()
            prob1 = np.prod(self.P_y(Y1))
            # Prior for choosing Y ^ w[t] -> (t)
            wprob1 = np.log(prob1) * w
            # P(x[i,t]==1|...) in Eq. 15. -> (i,t) [or (i,1) if w is scalar]
            prob2 = self.P_x_YZ(Y, Z)
            # P(x[i,t]|...) ^ w[:,t] -> (i,t)
            wprob2 = np.log(bernoulli.pmf(X, p=prob2)) * (
                        w[np.newaxis, :])
            # prod_i P(X[i,t]|Y[i,t],Z)P(Y[i,t]) -> (t)
            prod = wprob1 + np.sum(wprob2, axis=0)
            # sum_Y1 P(X|Y1,Z)P(Y1) -> (t)
            cum += np.exp(prod)
        # P(X|Y,Z)P(Y) -> ()
        ret = prob0 + np.sum(np.log(cum))
        return ret

    @staticmethod
    def m(Z, keepdims=True):
        '''Count out-edges of cause k'''
        # m[k] -> (k)
        m = np.sum(Z, axis=0, keepdims=keepdims)
        return m

    @staticmethod
    def m_(Z):
        '''Count out-edges of cause k omitting observable i'''
        # m[k] -> (k)
        m = BernoulliBetaAssumption.m(Z)
        # m[(not i),k] -> (i,k)
        m_ = m - Z
        return m_

    @staticmethod
    def K_new(Z):
        '''Count the columns of Z which contain a 1 only in row i'''
        # number of 1-s in each column of Z -> (1,k)
        column_loading = np.sum(Z, axis=0, keepdims=True)
        # zero the columns containing more than one 1-s -> (i,k)
        singles_columns = np.logical_and(Z, column_loading==1)
        # number of columns which contain a 1 only in row i -> (i,1)
        ret = np.sum(singles_columns.astype(int), axis=1, keepdims=True)
        return ret

    def get_theta(self):
        '''Measure the estimated value of theta'''
        # m[k] -> (k)
        m = BernoulliBetaAssumption.m(self.Z)
        # alpha/K
        AperK = self.alpha / self.K_effective
        #
        theta = (m + AperK) / float(self.N + AperK)
        return theta

    # Eq. 10 resulting from the prior on Z
    def P_z_Z(self, a):
        '''Prior z[i,k]==a conditional (K-vector or scalar) a and previous Z,
           i.e., z[(not i),k]'''
        assert np.all((a==1)|(a==0))
        # variables, alpha must be float, K effective might be inf
        Z = self.Z
        N = self.N
        # alpha/K
        AperK = self.alpha / self.K_effective
        # m[(not i),k] -> (i,k)
        m_ = BernoulliBetaAssumption.m_(Z).astype(float)
        # theta[k] are vectors -> is a matrix (i,k)
        theta = (m_ + AperK) / (N + AperK)
        # theta if a==1 else 1-theta -> (i,k)
        ret = bernoulli.pmf(a, p=theta)
        return ret

    # Eq. 11 used for updating Z
    def logP_z_XYZ(self, a):
        '''Enumerate the log posterior: element of Z equals binary scalar a'''
        assert a in [0,1]
        # variables
        X = self.X
        Y = self.Y
        Z = self.Z
        # theta[k]^a(1-theta[k])^(1-a) -> (i,k)
        prob1 = self.P_z_Z(a)
        # Z[i,k]@Y[k,t] -> (i,t)
        pure_scalar_prod = np.dot(Z[:,:],Y[:,:])
        # (a-Z[i,k,:])*Y[k,t] -> (i,k,t)
        correction = (a-Z[:,:,np.newaxis])*Y[:,:]
        # ... -> (i,k,t)
        conditional_scalar_prod = pure_scalar_prod[:,np.newaxis,:] + correction
        # similar to Eq. 3, to be multiplied along t -> (i,k,t)
        prob2 = 1 - np.power(1-self.lamb,conditional_scalar_prod) * (1-self.eps)
        # prob2 ^ w, to be multiplied along t -> (i,k,t)
        wprob2 = np.log(bernoulli.pmf(X[:,np.newaxis,:],p=prob2)) * (
                    self.w[np.newaxis, np.newaxis, :])
        #wprob2 = np.log(prob2) * self.w[...] was wrong in the paper
        # prob1 * prod_t(P(X[i,:,t]|prob2)) -> (i,k)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ret = np.log(prob1[:,:]) + np.sum(wprob2,axis=-1)
        return ret

    # Analogue to Eq. 10 resulting from the prior on Y
    def P_y(self, a):
        '''Prior y[k,t]==a conditional binary (T-vector or scalar) a'''
        assert np.all((a==1)|(a==0))
        # p
        p = self.p
        # p if a==1 else 1-p
        ret = bernoulli.pmf(a,p=p)
        return ret

    # Eq. 12 used for updating Y
    def logP_y_XYZ(self, a):
        '''Enumerate the log posterior: element of Y equals binary scalar a'''
        assert a in [0,1]
        X = self.X
        Y = self.Y
        Z = self.Z
        # p^a(1-p)^(1-a) -> ()
        prob1 = self.P_y(a)
        # Z[i,k]@Y[k,t] -> (i,t)
        pure_scalar_prod = np.dot(Z[:,:],Y[:,:])
        # Z[i,k,:]*(a-Y[k,t]) -> (i,k,t)
        correction = Z[:,:,np.newaxis]*(a-Y[:,:])
        # ... -> (i,k,t)
        conditional_scalar_prod = pure_scalar_prod[:,np.newaxis,:] + correction
        # to be multiplied along i -> (i,k,t)
        prob2 = 1 - np.power(1-self.lamb,conditional_scalar_prod) * (1-self.eps)
        # prob2 ^ w, to be multiplied along i -> (i,k,t)
        wprob2 = np.log(bernoulli.pmf(X[:,np.newaxis,:],p=prob2)) * (
                    self.w[np.newaxis, np.newaxis, :])
        #wprob2 = np.log(prob2) * self.w[...] was wrong in the paper
        # prob1 * prod_i(P(X[i,:,t]|prob2)) -> (k,t)
        ret = np.log(prob1) + np.sum(wprob2,axis=0)
        return ret

    # Eq. 12 modified for P(US|CS,...)
    def logP_y_XZ(self, v, X=None, Z=None, w=None, given_i=slice(None)):
        '''Enumerate all Y considering only features in given_i and conditional
           on the same binary vector v used as columns of Y for all trials'''
        # In this function, only X has trial-specific details
        # If unknown entries of X, i.e. given_i==false elements,
        # are set to the same value then it should give as good
        # results as omitting them as far as we can condition on
        # zero-measure sets (therefore we rather implement this
        # function for P=0 weight cases)
         
        X_self = X is None
        if X is None:
            X = self.X
        N, T = X.shape
        if Z is None:
            Z = self.Z
        N_, K = Z.shape
        assert N == N_

        if w is None:
            if X_self:
                w = self.w
            else:
                w = self.amo.calc_logw(self.amo.calc_loga(X.shape[1]))
        w = np.array(w, ndmin=1)
        assert (w.ndim == 1) and ((len(w)==1) or len(w)==T)

        assert np.all((v==1)|(v==0))
        assert len(v) == K
        
        if type(given_i) is not slice:
            # conversion is required for boolean slicing
            given_i = np.array(given_i)
            assert (given_i.ndim == 1) and (len(given_i) == N)

        # p^a(1-p)^(1-a) -> (k)
        prob1 = self.P_y(v)
        # Z[i,k]@v[k,:] -> (i,1)
        pure_scalar_prod = np.dot(Z[:,:],v[:,np.newaxis])
        # to be multiplied along i -> (i,1)
        prob2 = 1 - np.power(1-self.lamb,pure_scalar_prod) * (1-self.eps)
        # prob2, to be multiplied along i -> (i,1)
        uprob2 = np.log(bernoulli.pmf(X[given_i,:],p=prob2[given_i,:]))
        # prod_k(prob1) * prod_i(P(X[i,t]|prob2)) -> (t)
        ret = np.sum(np.log(prob1)) + np.sum(uprob2,axis=0)
        return ret

    # Eq. 14 (including Eq. 15)
    def logP_X_YZK(self, K_new):
        '''Likelihood of X conditional on K, Y and Z'''
        # K_new must be a vector containing sampled integer values
        p = self.p
        X = self.X
        Y = self.Y
        Z = self.Z
        # (1-lambda)^(Z[i,k]@Y[k,t]) -> (i,t)
        eta = np.power(1-self.lamb,np.dot(Z[:,:],Y[:,:]))
        # P(x[i,t]==1|...) in Eq. 15. -> (sample,i,t)
        prob = 1 - (1-self.eps) * eta * np.power(1-self.lamb*p,
                                                K_new[:,np.newaxis,np.newaxis])
        self.px = prob[0]
        # P[sample](x[i,t]==1|...) ^ w[:,:,t] -> (sample,i,t)
        wprob = np.log(bernoulli.pmf(X, p=prob)) * (
                    self.w[np.newaxis, np.newaxis, :])
        # P(X[i,:]|...) in Eq. 14. -> (sample,i)
        ret = np.sum(wprob, axis=-1)
        return ret

    # Eq. 13
    def logP_K_XYZ(self, K_new):
        '''Posterior of K conditional on Y and Z'''
        # K_new must be a vector containing sampled integer values
        alpha = self.alpha
        N = self.N
        # term 1 -> (sample,i)
        log_prob1 = self.logP_X_YZK(K_new)
        # term 2 -> (sample,1)
        log_prob2 = poisson.logpmf(K_new[:,np.newaxis], mu=alpha/N)
        #
        ret = log_prob1 + log_prob2
        return ret

    # Algo 2, lines 3-10
    def __Gibbs_sample_Z(self):
        '''Sample Z'''
        # m[(not i),k] -> (i,k)
        m_ = BernoulliBetaAssumption.m_(self.Z)
        # use P(z[i,k]=a|X,Y,Z[(not i,k)]) -> (i,k)
        Pz = p0_per_sum_p(logp=[self.logP_z_XYZ(1), self.logP_z_XYZ(0)])
        # sample z[i,k]
        try:
            z = bernoulli.rvs(p=Pz)
        except:
            print (Pz)
            raise
        # zero it where m[(not i),k]
        z[m_==0] = 0
        self.Z = z

    # Algo 2, lines 11-12
    def __Gibbs_sample_K(self):
        '''Sample K'''
        # maximum allowed value of new causes
        K_new_max = 10
        # sample values for K (we need to constrain it to remain tractable)
        K_new = np.arange(K_new_max).astype(int)
        # P(K_new[i]|...) -> (sample,i)
        PK = normalize_p(logp=self.logP_K_XYZ(K_new),axis=0)
        # sample K
        def choose(p):
            try:
                return np.random.choice(K_new, 1, p=p)
            except:
                print(p)
                raise
        K = np.apply_along_axis(choose, 0, PK)
        # total number of new columns to be added
        K_add = np.sum(K)
        # make room for them
        Z = np.hstack(( self.Z, np.zeros((self.N,K_add)) ))
        Y = np.vstack(( self.Y, np.zeros((K_add,self.T)) ))
        # put the ones
        K_sum = np.hstack(([0],np.cumsum(K)))
        for i in range(0,self.N):
            Z[i,K_sum[i]:K_sum[i+1]] = 1
        # store
        self.Y, self.Z = Y, Z

    # Algo 2, lines 13-15 and 18-20
    def __Gibbs_sample_Y(self):
        '''Sample Y'''
        # use P(y[k,t]=a|X,Y,Z[(not k,t)]) -> (k,t)
        Py = p0_per_sum_p(logp=[self.logP_y_XYZ(1), self.logP_y_XYZ(0)])
        # sample y[k,t]
        y = bernoulli.rvs(p=Py)
        # store
        self.Y = y

    # Algo 2, removal
    def __Gibbs_removal(self):
        '''Remove unused causes'''
        Y, Z = self.Y, self.Z
        m = BernoulliBetaAssumption.m(Z, keepdims=False)
        # keep colmns with m[k]>0 only
        keep = m>0
        # do not delete all columns, it won't work
        # FIXME: at some point Y still might get reduced to 1D
        if ~np.any(keep):
            keep[0]=True
        # Z[i,k]
        Z = np.compress(keep, Z, axis=1)
        # Y[k,t]
        Y = np.compress(keep, Y, axis=0)
        # store
        self.Y, self.Z = Y, Z

    # Algo 2
    def Gibbs_iterate(self):
        '''Iterate 1 cycle'''
        self.__Gibbs_sample_Z()
        #print ('N', self.N, 'K', self.K, 'T', self.T)
        self.__Gibbs_sample_K()
        #print ('N', self.N, 'K', self.K, 'T', self.T)
        self.__Gibbs_sample_Y()
        #print ('N', self.N, 'K', self.K, 'T', self.T)
        self.__Gibbs_removal()
        #print ('N', self.N, 'K', self.K, 'T', self.T)
