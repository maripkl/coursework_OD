import numpy as np

import scipy.linalg as spl


#  [gamma pi Li] =
def VB_diarization(X, m, iE, V, pi=None, gamma=None, maxSpeakers=10, maxIters=10, epsilon=1e-4, loopProb=0.99,
                   alphaQInit=1.0, ref=None, plot=False, minDur=1, Fa=1.0, Fb=1.0, return_clusters=False):

    """
    This is a simplified version of speaker diarization described in:
    Diez. M., Burget. L., Landini. F., Cernocky. J.
    Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors
    Variable names and equation numbers refer to those used in the paper
    Inputs:
    X           - T x D array, where columns are D dimensional feature vectors for T frames
    m           - C x D array of GMM component means, INCORRECT - !!!
    iE          - C x D array of GMM component inverse covariance matrix diagonals, INCORRECT - !!!
    V           - R x C x D array of eigenvoices, INCORRECT - !!!
    pi          - speaker priors, if any used for initialization
    gamma       - frame posteriors, if any used for initialization
    maxSpeakers - maximum number of speakers expected in the utterance
    maxIters    - maximum number of algorithm iterations
    epsilon     - stop iterating, if obj. fun. improvement is less than epsilon
    loopProb    - probability of not switching speakers between frames
    alphaQInit  - Dirichlet concentration parameter for initializing gamma
    ref         - T dim. integer vector with reference speaker ID (0:maxSpeakers)
                  per frame
    plot        - if set to True, plot per-frame speaker posteriors.
    minDur      - minimum number of frames between speaker turns imposed by linear
                  chains of HMM states corresponding to each speaker. All the states
                  in a chain share the same output distribution
    Fa          - scale sufficient statistics collected using UBM
    Fb          - speaker regularization coefficient Fb (controls final # of speaker)
    Outputs:
    gamma  - S x T matrix of posteriors attribution each frame to one of S possible
             speakers, where S is given by opts.maxSpeakers
    pi - S dimensional column vector of ML learned speaker priors. Ideally, these
         should allow to estimate # of speaker in the utterance as the
         probabilities of the redundant speaker should converge to zero.
    Li - values of auxiliary function (and DER and frame cross-entropy between gamma
         and reference if 'ref' is provided) over iterations.
    """

    nframes, D = X.shape  # feature dimensionality
    R = V.shape[0]  # subspace rank

    if pi is None:
        pi = np.ones(maxSpeakers)/maxSpeakers
    else:
        maxSpeakers = len(pi)

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(nframes, maxSpeakers))
        gamma = gamma / gamma.sum(1, keepdims=True)

    # calculate UBM mixture frame posteriors (i.e. per-frame zero order statistics)
    G = -0.5 * (np.sum((X-m).dot(iE)*(X-m), axis=1) - logdet(iE) + D*np.log(2*np.pi))
    #print("G", G[:5])
    LL = np.sum(G)  # total log-likelihood as calculated using UBM
    #print("LL", LL)
    
    VtiEV = V.dot(iE).dot(V.T)
    #print("VtiEV", VtiEV)
    #VtiEF = (X-m).dot(iE.dot(V).T) # (T,R) - BUG!
    VtiEF = (X-m).dot(iE.dot(V.T)) # (T,R)
    #print("VtiEF", VtiEF[:3, :])

    Li = [[LL*Fa]]  # for the 0-th iteration,
    if ref is not None:
        Li[-1] += [DER(gamma, ref), DER(gamma, ref, xentropy=True)]

    tr = np.eye(minDur*maxSpeakers, k=1)
    ip = np.zeros(minDur*maxSpeakers)
    for i in range(maxIters):
        L = 0  # objective function (37) (i.e. VB lower-bound on the evidence)
        Ns = np.sum(gamma, axis=0)[:, np.newaxis, np.newaxis]  # bracket in eq. (34) for all 's'
        VtiEFs = gamma.T.dot(VtiEF)[:, :, np.newaxis]  # eq. (35) except for \Lambda_s^{-1} for all 's'
        invLs = np.linalg.inv(np.eye(R)[np.newaxis, :, :] + Ns * VtiEV[np.newaxis, :, :] * Fa / Fb)  # eq. (34) inverse
        a = np.matmul(invLs, VtiEFs).squeeze(axis=-1) * Fa / Fb  # eq. (35)
        # eq. (29) except for the prior term \ln \pi_s. Our prior is given by HMM
        # transition probability matrix. Instead of eq. (30), we need to use
        # forward-backward algorithm to calculate per-frame speaker posteriors,
        # where 'lls' plays role of HMM output log-probabilities
        lls = Fa * (
            G[:, np.newaxis] + VtiEF.dot(a.T) - 0.5 * (
                (invLs+np.matmul(a[:, :, np.newaxis], a[:, np.newaxis, :])) * VtiEV[np.newaxis, :, :]
            ).sum(axis=(1, 2))
        ) # (T, K)

        for sid in range(maxSpeakers):
            L += Fb * 0.5 * (logdet(invLs[sid]) - np.sum(np.diag(invLs[sid]) + a[sid]**2, 0) + R)
            
        #print(i, L)

        # Construct transition probability matrix with linear chain of 'minDur'
        # states for each of 'maxSpeaker' speaker. The last state in each chain has
        # self-loop probability 'loopProb' and the transition probabilities to the
        # initial chain states given by vector '(1-loopProb) * pi'. From all other,
        # states, one must move to the next state in the chain with probability one.
        tr[minDur-1::minDur, 0::minDur] = (1-loopProb) * pi
        tr[(np.arange(1, maxSpeakers+1) * minDur - 1,) * 2] += loopProb
        ip[::minDur] = pi
        # per-frame HMM state posteriors. Note that we can have linear chain of minDur states
        # for each speaker.
        gamma, tll, lf, lb = forward_backward(lls.repeat(minDur, axis=1), tr, ip)

        # Right after updating q(Z), tll is E{log p(X|,Y,Z)} - KL{q(Z)||p(Z)}.
        # L now contains -KL{q(Y)||p(Y)}. Therefore, L+ttl is correct value for ELBO.
        L += tll
        Li.append([L])

        # ML estimate of speaker prior probabilities (analogue to eq. (38))
        with np.errstate(divide="ignore"):  # too close to 0 values do not change the result
            pi = gamma[0, ::minDur] + np.exp(
                logsumexp(lf[:-1, minDur-1::minDur], axis=1)[:, np.newaxis]
                + lb[1:, ::minDur] + lls[1:] + np.log((1-loopProb) * pi) - tll
            ).sum(axis=0)
        pi = pi / pi.sum()

        # per-frame speaker posteriors (analogue to eq. (30)), obtained by summing
        # HMM state posteriors corresponding to each speaker
        gamma = gamma.reshape(len(gamma), maxSpeakers, minDur).sum(axis=2)

        # if reference is provided, report DER, cross-entropy and plot the figures
        if ref is not None:
            Li[-1] += [DER(gamma, ref), DER(gamma, ref, xentropy=True)]

            if plot:
                import matplotlib.pyplot
                if i == 0:
                    matplotlib.pyplot.clf()
                matplotlib.pyplot.subplot(maxIters, 1, i+1)
                matplotlib.pyplot.plot(gamma, lw=2)
                matplotlib.pyplot.imshow(np.atleast_2d(ref), interpolation='none', aspect='auto',
                                         cmap=matplotlib.pyplot.cm.Pastel1, extent=(0, len(ref), -0.05, 1.05))
            print(i, Li[-2])

        if i > 0 and L - Li[-2][0] < epsilon:
            if L - Li[-1][0] < 0:
                print('WARNING: Value of auxiliary function has decreased!')
            break
            
    if return_clusters:
        means = m + np.dot(a, V) # (K, D)
        covs = np.zeros((maxSpeakers, D, D))
        for k in range(maxSpeakers):
            covs[k, :, :] = V.T.dot(invLs[k, :, :]).dot(V) + np.eye(D)
        return gamma, pi, Li, means, covs

    return gamma, pi, Li



###############################################################################
# Module private functions
###############################################################################
def logsumexp(x, axis=0):
    xmax = x.max(axis)
    with np.errstate(invalid="ignore"):  # nans do not affect inf
        x = xmax + np.log(np.sum(np.exp(x - np.expand_dims(xmax, axis)), axis))
    infs = np.isinf(xmax)
    if np.ndim(x) > 0:
        x[infs] = xmax[infs]
    elif infs:
        x = xmax
    return x


def logdet(A):
    return 2*np.sum(np.log(np.diag(spl.cholesky(A))))


def forward_backward(lls, tr, ip):
    """
    Inputs:
        lls - matrix of per-frame log HMM state output probabilities
        tr  - transition probability matrix
        ip  - vector of initial state probabilities (i.e. statrting in the state)
    Outputs:
        pi  - matrix of per-frame state occupation posteriors
        tll - total (forward) log-likelihood
        lfw - log forward probabilities
        lfw - log backward probabilities
    """
    with np.errstate(divide="ignore"):  # too close to 0 values do not change the result
        ltr = np.log(tr)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    with np.errstate(divide="ignore"):  # too close to 0 values do not change the result
        lfw[0] = lls[0] + np.log(ip)
    lbw[-1] = 0.0

    for i in range(1, len(lls)):
        lfw[i] = lls[i] + logsumexp(lfw[i-1] + ltr.T, axis=1)

    for i in reversed(range(len(lls)-1)):
        lbw[i] = logsumexp(ltr + lls[i+1] + lbw[i+1], axis=1)

    tll = logsumexp(lfw[-1])
    pi = np.exp(lfw + lbw - tll)
    return pi, tll, lfw, lbw



#  [gamma pi Li] =
def VB_diarization_UP(X, m, iE, V, pi=None, gamma=None, maxSpeakers=10, maxIters=10, epsilon=1e-4, loopProb=0.99,
                   alphaQInit=1.0, ref=None, plot=False, minDur=1, Fa=1.0, Fb=1.0, return_clusters=False):

    """
    This is a simplified version of speaker diarization described in:
    Diez. M., Burget. L., Landini. F., Cernocky. J.
    Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors
    Variable names and equation numbers refer to those used in the paper
    Inputs:
    X           - T x D array, where columns are D dimensional feature vectors for T frames
    m           - 1 x D array, mean
    iE          - T x D array inverse covariance matrix diagonals for T frames
    V           - R x D array of eigenvoices
    pi          - speaker priors, if any used for initialization
    gamma       - frame posteriors, if any used for initialization
    maxSpeakers - maximum number of speakers expected in the utterance
    maxIters    - maximum number of algorithm iterations
    epsilon     - stop iterating, if obj. fun. improvement is less than epsilon
    loopProb    - probability of not switching speakers between frames
    alphaQInit  - Dirichlet concentration parameter for initializing gamma
    ref         - T dim. integer vector with reference speaker ID (0:maxSpeakers)
                  per frame
    plot        - if set to True, plot per-frame speaker posteriors.
    minDur      - minimum number of frames between speaker turns imposed by linear
                  chains of HMM states corresponding to each speaker. All the states
                  in a chain share the same output distribution
    Fa          - scale sufficient statistics collected using UBM
    Fb          - speaker regularization coefficient Fb (controls final # of speaker)
    Outputs:
    gamma  - S x T matrix of posteriors attribution each frame to one of S possible
             speakers, where S is given by opts.maxSpeakers
    pi - S dimensional column vector of ML learned speaker priors. Ideally, these
         should allow to estimate # of speaker in the utterance as the
         probabilities of the redundant speaker should converge to zero.
    Li - values of auxiliary function (and DER and frame cross-entropy between gamma
         and reference if 'ref' is provided) over iterations.
    """

    nframes, D = X.shape  # feature dimensionality
    R = V.shape[0]  # subspace rank

    if pi is None:
        pi = np.ones(maxSpeakers)/maxSpeakers
    else:
        maxSpeakers = len(pi)

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(nframes, maxSpeakers)) # (T, K)
        gamma = gamma / gamma.sum(1, keepdims=True)

    # calculate UBM mixture frame posteriors (i.e. per-frame zero order statistics)
    G = -0.5 * (np.sum((X-m)*iE*(X-m), axis=1) - np.sum(np.log(iE), axis=1) + D*np.log(2*np.pi))
    #print("G", G[:5])
    LL = np.sum(G)  # total log-likelihood as calculated using UBM
    #print("LL", LL)
    
    #print('!', iE[:, :, np.newaxis].shape, V.T[np.newaxis, :, :].shape)
    sqriEV = np.sqrt(iE[:, :, np.newaxis]) * V.T[np.newaxis, :, :] # (T, D, 1) * (1, D, R) = (T, D, R)
    #print(sqriEV.shape)
    VtiEV = np.matmul(sqriEV.transpose((0, 2, 1)), sqriEV) # (T, R, R)
    #print(VtiEV.shape)
    
    #VtiEV = V.dot(iE).dot(V.T) # 
    #VtiEF = (X-m).dot(iE.dot(V).T) # (T,R) - BUG!
    VtiEF = ((X-m) * iE).dot(V.T) # (T, R)
    
    #print("VtiEV", VtiEV[:3, :]) # 
    #print("VtiEF", VtiEF[:3, :])
    
    #print(G.shape, LL)
    #print(VtiEV.shape, VtiEF.shape) # (d,d), (T,d)

    Li = [[LL*Fa]]  # for the 0-th iteration,

    tr = np.eye(minDur*maxSpeakers, k=1)
    ip = np.zeros(minDur*maxSpeakers)
    for i in range(maxIters):
        L = 0  # objective function (37) (i.e. VB lower-bound on the evidence)
        
        #Ns = np.sum(gamma, axis=0)[:, np.newaxis, np.newaxis]  # bracket in eq. (34) for all 's', (K, 1, 1)
        
        VtiEFs = gamma.T.dot(VtiEF)  # eq. (35) except for \Lambda_s^{-1} for all 's', (K, R)
        VtiEFs = VtiEFs[:, :, np.newaxis] # (K, R, 1)
        I = np.eye(R)[np.newaxis, :, :] # (1, R, R)
        gammaVtiEV = np.tensordot(gamma, VtiEV, (0, 0))
        # (K, R, R)
        invLs = np.linalg.inv(I + gammaVtiEV * Fa / Fb)  # eq. (34) inverse, (K, R, R)
        a = np.matmul(invLs, VtiEFs).squeeze(axis=-1) * Fa / Fb  # eq. (35) # (K, R)
        

        # eq. (29) except for the prior term \ln \pi_s. Our prior is given by HMM
        # transition probability matrix. Instead of eq. (30), we need to use
        # forward-backward algorithm to calculate per-frame speaker posteriors,
        # where 'lls' plays role of HMM output log-probabilities
        
        #tmp = ((invLs + np.matmul(a[:, :, np.newaxis], a[:, np.newaxis, :])) * VtiEV[0,:,:][np.newaxis, :, :]).sum(axis=(1, 2))
        #print(G[:, np.newaxis].shape, VtiEF.dot(a.T).shape, tmp.shape)
        
        # (T, 1) + (T, K) - 0.5 * (K, R, R).sum((1,2))
        
        # (T, 1) + (T, K) - 0.5 * (K, R, R).sum((1,2))
        
        invLs_aa = (invLs + np.matmul(a[:, :, np.newaxis], a[:, np.newaxis, :]))
        
        #print(np.tensordot(VtiEV, invLs_aa, axes=([1,2],[1,2])).shape)
        lls = Fa * (
            G[:, np.newaxis] + VtiEF.dot(a.T) - 0.5 * np.tensordot(VtiEV, invLs_aa, axes=([1,2],[1,2]))) # (T, K)
        

        for sid in range(maxSpeakers):
            L += Fb * 0.5 * (logdet(invLs[sid]) - np.sum(np.diag(invLs[sid]) + a[sid]**2, 0) + R)
            
        #print(i, L)

        # Construct transition probability matrix with linear chain of 'minDur'
        # states for each of 'maxSpeaker' speaker. The last state in each chain has
        # self-loop probability 'loopProb' and the transition probabilities to the
        # initial chain states given by vector '(1-loopProb) * pi'. From all other,
        # states, one must move to the next state in the chain with probability one.
        tr[minDur-1::minDur, 0::minDur] = (1-loopProb) * pi
        tr[(np.arange(1, maxSpeakers+1) * minDur - 1,) * 2] += loopProb
        ip[::minDur] = pi
        # per-frame HMM state posteriors. Note that we can have linear chain of minDur states
        # for each speaker.
        gamma, tll, lf, lb = forward_backward(lls.repeat(minDur, axis=1), tr, ip)

        # Right after updating q(Z), tll is E{log p(X|,Y,Z)} - KL{q(Z)||p(Z)}.
        # L now contains -KL{q(Y)||p(Y)}. Therefore, L+ttl is correct value for ELBO.
        L += tll
        Li.append([L])

        # ML estimate of speaker prior probabilities (analogue to eq. (38))
        with np.errstate(divide="ignore"):  # too close to 0 values do not change the result
            pi = gamma[0, ::minDur] + np.exp(
                logsumexp(lf[:-1, minDur-1::minDur], axis=1)[:, np.newaxis]
                + lb[1:, ::minDur] + lls[1:] + np.log((1-loopProb) * pi) - tll
            ).sum(axis=0)
        pi = pi / pi.sum()

        # per-frame speaker posteriors (analogue to eq. (30)), obtained by summing
        # HMM state posteriors corresponding to each speaker
        gamma = gamma.reshape(len(gamma), maxSpeakers, minDur).sum(axis=2)

        if i > 0 and L - Li[-2][0] < epsilon:
            if L - Li[-1][0] < 0:
                print('WARNING: Value of auxiliary function has decreased!')
            break
    
    if return_clusters:
        means = m + np.dot(a, V) # (K, D)
        covs = np.zeros((maxSpeakers, D, D))
        for k in range(maxSpeakers):
            covs[k, :, :] = V.T.dot(invLs[k, :, :]).dot(V) + np.eye(D)
        return gamma, pi, Li, means, covs

    return gamma, pi, Li
