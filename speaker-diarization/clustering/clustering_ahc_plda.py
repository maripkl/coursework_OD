import numpy as np
from scipy.special import gammaln, psi
from numpy.random import choice
from scipy.special import expit as sigmoid
from scipy.special import logit
from scipy.optimize import minimize, minimize_scalar

#from lib.special.softplus import softplus, softplusinv
#from lib.combin.partitions import partitions_and_subsets

class CRP:
    def __init__(self, alpha, beta):
        assert alpha >= 0 <= beta <= 1
        self.alpha = alpha
        self.beta = beta
        
    def sample(self,n):
        alpha, beta = self.alpha, self.beta
        labels = np.zeros(n,int)  # restricted growth string, labels start at 0   
        counts = np.zeros(n,int)  # table occupancies (up to n tables)
        p = np.empty(n)
        counts[0] = 1             # seat first customer at table 0
        nt = 1                    # number of occupied tables  
        for i in range(1,n):      # seat rest of customers
            # i is number of seated customers and index of to-be-seated customer
            pi = p[:nt+1]
            pi[:nt] = (counts[:nt] - beta) / (i + alpha) # occupied tables
            pi[nt] = (alpha + nt*beta ) / (i + alpha)    # new table
            t = choice(nt+1, None, True, pi)             # chosen table
            labels[i] = t
            counts[t] += 1
            if t == nt: nt += 1                          # new table was chosen
        return labels, counts[:nt]    
    
    def samples(self,n,m):
        """
        Sample m independent partitions of size n from this CRP
        Returns a list of m arrays of block sizes.
        (The array sizes are variable, depending on the number of blocks in 
        the partition.)
        """
        assert n >= 1 <= m
        counts_list = []
        for i in range(m):
            labels, counts = self.sample(n)
            counts_list.append(counts)
        return counts_list
    
    def logprob(self,counts):
        """
        Returns log P(labels | crp), where labels is represented by the given
        table occupancy counts.
        """
        alpha, beta = self.alpha, self.beta
        
        if alpha == np.Inf and beta==1: #singleton tables
            return 0.0 if all(counts==1) else -np.Inf
        
        if alpha==0 and beta==0:       #single table
            return 0.0 if len(counts)==1 else -np.Inf
        
        if alpha>0 and beta>0:  # general case (2 parameter Pitman-Yor CRP)
            return logprob_alpha_beta(alpha,beta,counts)
  
        if beta==0 and alpha>0:  # classical 1-parameter CRP
           return logprob_alpha(alpha,counts)
       
        if beta>0 and alpha==0:
            return logprob_beta(beta,counts)
        
        assert False
        
    def llr_joins(self,counts,i):
        """
        Let logLR(i,j) = log P(join(i,j)|crp) - log P( labels| crp), where
        labels is represented by the given occupancy counts; and where join(i,j)
        joins tables i and j, while leaving other tables as-is. A vector is 
        returned with all logLR(i,j), with j > i. 
        
        For use by AHC (agglomerative hierarchical clustering) algorithms that 
        seek greedy MAP partitions, where this CRP forms the partition prior.
        """
        alpha, beta = self.alpha, self.beta
        K = len(counts)  # tables
        assert K > 1
        ci = counts[i]
        cj = counts[i+1:]
        llr = gammaln(1-beta) - np.log(beta) - np.log(alpha/beta + K-1) 
        llr += gammaln(cj+(ci-beta)) - gammaln(ci-beta) - gammaln(cj-beta) 
        return llr
            

    
    def exp_num_tables(self,n):
        """
        n: number of customers
        """
        alpha, beta = self.alpha, self.beta
        if alpha==0 and beta==0:
            e = 1
        elif alpha == np.Inf:
            e = n
        elif alpha>0 and beta>0:      
            A = gammaln(alpha + beta + n) + gammaln(alpha + 1) \
                - np.log(beta) - gammaln(alpha+n) - gammaln(alpha+beta)
            B = alpha/beta
            e = B*np.expm1(A-np.log(B))   # exp(A)-B
        elif alpha>0 and beta==0:
            e = alpha*( psi(n+alpha) - psi(alpha) )
        elif alpha==0 and beta>0:
            A = gammaln(beta + n) - np.log(beta) - gammaln(n) - gammaln(beta)
            e = np.exp(A)
        return e
    
    def __repr__(self):
        return f"CRP(alpha={self.alpha}, beta={self.beta})"
    
    
#     def logprobtable(self,P,S=None):
#         """
#         Returns pre-computed table of log-probabilities, for every partition
#         of a set of n elements. 
        
#         Usage:
            
#             P, S = lib.combin.partitions_and_subsets(n,dtype=bool)
#             table = crp.logprobtable(P,S)
        
#         or, equivalently:
            
#             table = crp.logprobtable(n)
        
#         """
#         if S is None: 
#             assert type(P)==int and P>0
#             P,S = partitions_and_subsets(P,dtype=bool)
#         counts = S.sum(axis=1)
#         Bn, ns = P.shape
#         L = [self.logprob(counts[P[i,:].astype(bool,copy=False)]) for i in range(Bn)]
#         return np.array(L)
    
    
    def ahc(self,labels):
        """
        Returns an AHC object, initialized at the given labels.
        
        For use by AHC (agglomerative hierarchical clustering) algorithms that 
        seek greedy MAP partitions, where this CRP forms the partition prior.
        
        """
        return AHC(self,labels)
    
    
    
class AHC:
    """
        For use by AHC (agglomerative hierarchical clustering) algorithms that 
        seek greedy MAP partitions, where this CRP forms the partition prior.
    """    
    def __init__(self,crp,labels):
        self.crp = crp
        tables, counts = np.unique(labels,return_counts=True)
        self.counts = counts
        
    def llr_joins(self,i):
        """
        Scores in logLR form, the CRP prior's contribution when joining tabls 
        i with all tables j > i.
        """
        crp, counts = self.crp, self.counts
        return crp.llr_joins(counts,i)
    
    def join(self,i,j):
        """
        Joins tables i and j in this AHC object.
        """
        counts = self.counts
        counts[i] += counts[j]
        self.counts = np.delete(counts,j)
        

        
class SingletonDict(dict):
    def __getitem__(self,key):
        return super().__getitem__(key) if key in self else {key}


class CluteringAHC:
    def __init__(self, w_inv, alpha, beta, X, B=None):
        if B is not None:
            assert X.shape == B.shape
        n,d = X.shape
        
        prior = CRP(alpha, beta)
        
        self.n = self.N = n
        if B is None:
            self.R = R = np.tile(w_inv,(n,1))
        else:
            self.R = R = (w_inv * B) / (w_inv + B)
        self.RX = RX = R*X         #(n,d)
        
        self.LLH = (RX**2/(1.0 + R) - np.log1p(R) ).sum(axis=1) / 2.0  #(n,)
        self.LLRs = []
        
        labels = np.arange(n, dtype=int) # full length labels, contains result
        self.ind = labels.copy() # 
        
        self.prior_ahc = prior.ahc(labels)
        
        # map every element to a singleton cluster containing that element
        self.clusters = SingletonDict()     
        
    def join(self,i,j):
        clusters = self.clusters
        join = clusters[i] | clusters[j]
        for e in join: clusters[e] = join
        
        
    def iteration(self, thr = 0.0):
        RX, R, n = self.RX, self.R, self.n
        prior_ahc, LLH = self.prior_ahc, self.LLH
        ind = self.ind
        
        #M = np.full((n,n),-np.Inf)
        
        maxval = -np.Inf
        for i in range(n-1):
            r = R[i,:]                    # (d,)      
            rR = r + R[i+1:,:]            # (n-i-1, d)
            rx = RX[i,:]
            rxRX = rx + RX[i+1:,:]
            llh = (rxRX**2/(1.0+rR) - np.log1p(rR) ).sum(axis=1) / 2.0  
            score = llh + prior_ahc.llr_joins(i) - LLH[i] - LLH[i+1:]
            #M[i,i+1:] = score
            j = score.argmax()
            scj = score[j]
            #print(i,i+j+1,': ',np.around(np.exp(scj),1))
            if scj > maxval:
                maxi = i
                maxj = j + i + 1
                maxval = scj
        
        #print(np.around(np.exp(M),1),'\n')
        LLRs = self.LLRs
        LLRs.append(maxval)         
          
        if maxval > thr:
            
            #print('joining: ',maxi,'+',maxj)
            #print('ind = ',ind)
            ii, jj = ind[maxi], ind[maxj]
            #print('joining: ',ii,'+',jj)
            self.join(ii,jj)
            
            
            RX[maxi,:] += RX[maxj,:]
            R[maxi,:] += R[maxj,:]        
            self.RX = np.delete(RX,maxj,axis=0)         
            self.R = np.delete(R,maxj,axis=0)

            self.n = n-1

            prior_ahc.join(maxi,maxj) 

            LLH[maxi] = maxval + LLH[maxi] + LLH[maxj]
            self.LLH = np.delete(LLH,maxj)

            self.ind = np.delete(ind,maxj)
        
        return maxval
    
    
    def cluster(self, thr = 0.0):
        while self.n > 1:
            llr = self.iteration(thr)
            if llr <= thr: break
        #return clusters2labels(self.clusters,self.N)
        return self.labelclusters()
    
    
    def labelclusters(self):
        clusters, n = self.clusters, self.N
        labels = np.full(n,-1)
        label = -1
        for i in range(n):
            s = clusters[i]
            for e in s: break #get first set element
            if labels[e] < 0: 
                label += 1
                labels[list(s)] = label
        return labels    
    