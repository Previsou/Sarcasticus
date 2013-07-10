from virony.data.preprocessing import LabelEncoder
from virony.parallel import parallel_map

from pandas import DataFrame
import time
cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
        
class UnaryFactor(object):
    
    def __init__(self, nb_states, nb_labels, weight, boundary, observations):

        self.K = nb_states
        self.G = nb_labels
        self.T = observations.shape[0]
        self.shift = 1 + self.G + boundary
        self.log = np.empty((self.G, self.T, self.K))
        self.values = np.empty((self.G, self.T, self.K))
        self.observations = observations
        self.scale = float(2*self.observations.nnz) / self.T
        self.scale = pow(1+self.scale, 2)
        
        self._computeValues(weight)
        
    def _computeValues(self, weight):

        for g in range(self.G):
            for s in range(self.K):
                abs_offset = 1 + self.shift * s
                select = slice(abs_offset+1+self.G, abs_offset+self.shift)
                temp = self.observations * weight[select]
                temp += weight[0] + weight[abs_offset + 1+g]
                temp += weight[abs_offset]/self.T* (1+np.arange(self.T))    #intercept and time dependence
                self.log[g,:,s] = temp / self.scale
        
        
        # Scaling
        for t in range(self.T):
            self.log[:,t] -= self.log[:,t].mean()
        self.values = np.nan_to_num(np.exp(self.log))
        
        
    def featureVector(self, glabel, time, state1):
        
        abs_offset = 1 + self.shift * state1
        rows, cols = self.observations.nonzero()
        cols += 1+self.G+abs_offset
        data = self.observations.data[rows==time]
            
        return abs_offset, cols[rows==time], data, self.scale

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

########################################################################
##########    Observation independent factor     #######################
########################################################################
class PairFactor(object):
    
    def __init__(self, nb_states, nb_labels, weight, boundary, observations):

        self.K = nb_states
        self.G = nb_labels
        self.T = observations.shape[0]
        self.log = np.empty((self.G, self.T, self.K, self.K))
        self.values = np.empty((self.G, self.T, self.K, self.K))
        self.shift = 1 + self.G * self.K
        self.scale = pow(2+self.K*self.G,2)
        
        self._computeValues(weight)
        
    def _computeValues(self, weight):   # to optimize
        
        
        for s2 in range(self.K):
            abs_offset = 1+ self.shift * s2
            temp = weight[0]
            for t in range(self.T):
                temp +=  weight[abs_offset] * float(t)/self.T # intercept, time dependency
                for g in range(self.G):
                    for s1 in range(self.K):
                        temp += weight[abs_offset+1+ g * self.K + s1]         # class and preceding state dependency
                        self.log[g, t, s1, s2] = temp / self.scale
                        
        # Scaling
        for t in range(1, self.T):
            self.log[:,t] -= self.log[:,t].mean()

        self.log[:,0] = 1.
        self.values = np.nan_to_num(np.exp(self.log))
    
            
    def featureVector(self, glabel, time, state1, state2):
        
        vect = np.zeros(1+self.K*self.shift, dtype=float)
        vect[0] = 1.0
        
        abs_offset = 1 + self.shift * state2
        vect[abs_offset] = float(time)/self.T
        vect[abs_offset+1+ glabel*self.K + state1] = 1.0
                
        return vect/self.scale

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

cdef class AbstractCRF(object):
    
    cdef:
        readonly unsigned int G, K, boundary
        object l_encoder, s_encoder
    
    
    def __init__(self, *args, **kwargs):
        pass
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef object forward_backward(self, np.ndarray[DTYPE_t, ndim=3] unary, np.ndarray[DTYPE_t, ndim=4] pair):
        
        #####   Forward     #####
        # Sequence length
        cdef unsigned int T = unary.shape[1]
        cdef double temp = 0.
        cdef double temp_scale
        cdef int g, t, a, s1, s2
    
        cdef np.ndarray[DTYPE_t, ndim=3] alpha = np.zeros((self.G, T, self.K), dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=3] beta = np.zeros((self.G, T, self.K), dtype=np.float64)
        cdef np.ndarray[DTYPE_t, ndim=1] scale = np.zeros(T, dtype=float)
        
        # Forward initialization
        for g in range(self.G):
            for a in range(self.K):
                temp = unary[g, 0, a]
                alpha[g, 0, a] = temp
                scale[0] += temp
                
        temp_scale = 1.0 / scale[0]
        for g in range(self.G):
            for a in range(self.K):
                alpha[g, 0, a] *= temp_scale
            
        # Forward propagation
        for t in range(1,T):
            for g in range(self.G):
                for s2 in range(self.K):
                    for s1 in range(self.K):
                        temp = pair[g, t, s1, s2] * alpha[g, t-1, s1] * unary[g, t, s2]
                        alpha[g, t, s2] += temp
                        scale[t] += temp
                        
            temp_scale = 1.0 / scale[t]
            for g in range(self.G):
                for a in range(self.K):
                    alpha[g,t,a] *= temp_scale
            
            
        
        # Backward initialization
        for g in range(self.G):
            for a in range(self.K):
                beta[g,T-1,a] = 1.0
        
        # Backward propagation
        for t in range(T-2, -1, -1):
            temp_scale = 1.0 / scale[t+1]
            for g in range(self.G):
                for s1 in range(self.K):
                    for s2 in range(self.K):
                        beta[g,t,s1] += beta[g, t+1, s2] * pair[g, t+1, s1, s2] * unary[g, t+1, s2]
                    beta[g,t,s1] *= temp_scale
                
        return alpha, beta, scale

    def observed(self, glabel, labels, types):
        
        encoded_labels = self.s_encoder.transform(labels)
        g_enc = self.l_encoder.transform([glabel])[0]
        temp = []
        
        if "unary" in types:
            mu_unary = np.zeros((self.G, len(labels), self.K))
            for t, enc_label in enumerate(encoded_labels):
                mu_unary[g_enc, t, enc_label] = 1.0
            temp.append(mu_unary)
            
        if "pair" in types:
            mu_pair = np.zeros((self.G, len(labels), self.K, self.K))
            gen = ((i, encoded_labels[i-1], encoded_labels[i]) for i in range(1, len(labels)))
            for t, a, b in gen:
                mu_pair[g_enc, t,a,b] = 1.0
            temp.append(mu_pair)
            
        return temp
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef object viterbi(self, np.ndarray[DTYPE_t, ndim=3] unary_log, np.ndarray[DTYPE_t, ndim=4] pair_log):
        
        # Sequence length
        cdef int T = unary_log.shape[1]
        cdef int g,t,s1,s2,s,arg
        cdef double proba, temp
        cdef np.ndarray[DTYPE_t, ndim=3] viterbi_proba = unary_log
        cdef np.ndarray[np.int_t, ndim=3] viterbi_path = np.empty((self.G, T, self.K), dtype=int)
        cdef np.ndarray[DTYPE_t, ndim=3] most_likely_sequences = np.zeros((self.G, T, self.K), dtype=float)
        cdef np.ndarray[DTYPE_t, ndim=1] best_proba = np.empty(self.G)
        
        # Initialization
        for g in range(self.G):
            for s in range(self.K):
                viterbi_path[g, 0, s] = s
        
        # Recurrence
        for g in range(self.G):
            for t in range(1,T):
                for s2 in range(self.K):
                    arg = 0
                    proba = viterbi_proba[g, t-1, 0]
                    for s1 in range(1,self.K):
                        temp = viterbi_proba[g, t-1, s1] + pair_log[g, t, s1, s2]
                        if temp > proba:
                            arg = s1
                            proba = temp
                    viterbi_proba[g, t, s2] += temp
                    viterbi_path[g, t, s2] = arg
                        
        # Extract Viterbi path
        for g in range(self.G):
        
            # Find argmax
            arg = 0
            proba = viterbi_proba[g, T-1, 0]
            for s1 in range(1,self.K):
                temp = viterbi_proba[g, T-1, s1]
                if temp > proba:
                    arg = s1
                    proba = temp
                    
            most_likely_sequences[g, T-1, arg] = 1.0
            best_proba[g] = proba
            
            for t in range(T-1, 0, -1):
                arg =  viterbi_path[g, t, arg]
                most_likely_sequences[g, t-1, arg] = 1.0
        
        return most_likely_sequences, best_proba


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef object viterbi_segmentation(self, np.ndarray[DTYPE_t, ndim=2] unary_log, np.ndarray[DTYPE_t, ndim=3] pair_log):
        
        # Sequence length
        cdef int T = unary_log.shape[0]
        cdef int t,s1,s2,s,arg
        cdef double proba, temp
        cdef np.ndarray[DTYPE_t, ndim=2] viterbi_proba = unary_log
        cdef np.ndarray[np.int_t, ndim=2] viterbi_path = np.empty((T, self.K), dtype=int)
        cdef np.ndarray[DTYPE_t, ndim=2] most_likely_sequence = np.zeros((T, self.K), dtype=float)
        
        # Initialization
        for s2 in range(self.K):
            viterbi_path[0, s2] = s2
        
        # Recurrence
        for t in range(1,T):
            for s2 in range(self.K):
                arg = 0
                proba = viterbi_proba[t-1, 0]
                for s1 in range(1,self.K):
                    temp = viterbi_proba[t-1, s1] + pair_log[t, s1, s2]
                    if temp > proba:
                        arg = s1
                        proba = temp
                viterbi_proba[t, s2] += temp
                viterbi_path[t, s2] = arg
                        
        # Extract Viterbi path
        temp = viterbi_proba[T-1].argmax()
        most_likely_sequence[T-1, temp] = 1.0
        for t in range(T-1, 0, -1):
            temp =  viterbi_path[t, temp]
            most_likely_sequence[t-1, temp] = 1.0
    
        return most_likely_sequence, viterbi_proba[T-1].max()
  
    def toScores(self, confusion_matrix, index):
        
        scores = DataFrame(None, index=index)
        scores["precision"] = [confusion_matrix[k,k] / confusion_matrix[k,:].sum() for k in range(self.K)]
        scores["recall"] = [confusion_matrix[k,k] / confusion_matrix[:,k].sum() for k in range(self.K)]
        scores["F1"] = 2* scores['precision'] * scores['recall'] / (scores['precision'] + scores['recall'])
        accuracy = sum(confusion_matrix[k,k] for k in range(self.K)) / confusion_matrix.sum()
        
        return scores, accuracy
    
    def expectation(self, train_meta, raw_data, semi_supervised):   # Parallel
        temp = [(entry, raw_data[entry["slice"]], semi_supervised) for entry in train_meta]
        return parallel_map(self.expectation_entry, temp, NUMCORES=7)
    
    def ifit(self, train_meta, train_data, test_meta, test_data, alphas, max_iter=10, semi_supervised=True, method="vSGD_fd"):
        
        start = time.time()
        EMiteration = 0
        
        yield self.training_report(train_meta, train_data, test_meta, test_data, alphas, start, EMiteration,0)
        
        while EMiteration < max_iter:
            EMiteration += 1
            # E step
            responsibilities = self.expectation(train_meta, train_data, semi_supervised)
            
            # M step
            if method == "vSGD_fd":
                SGDiteration = self.vSGD_fd(responsibilities, train_meta, train_data, alphas)
            else:
                SGDiteration = self.sgd(responsibilities, train_meta, train_data, alphas, EMiteration)
            
            # Report
            yield self.training_report(train_meta, train_data , test_meta, test_data, alphas, start, EMiteration, SGDiteration)
       
    def training_report(self, train_meta, train_data , test_meta, test_data, alphas, start, EMiteration, SGDiteration):
            
        scores = self.test_error(test_meta, test_data)
        neg_log_like = self.neg_log_likelihood(train_meta, train_data, alphas)
        summary = { "EMiteration":EMiteration, "SGDiteration":SGDiteration, "elapsed": time.time()-start, 
                "neg log likelihood":neg_log_like,
                "norm_w_unary":abs(self.w_unary**2).sum()}

        summary["sent_scores"] =  scores[0]
        if len(scores)==2:
            summary["document_scores"] =  scores[1]
        
        return summary

    def sgd(self, responsibilities, train_meta, raw_data, alphas=(1e-5,1e-5)):
    
        t = 0.1   # Assumed we have already seen 10 training samples
        for epoch in range(5):
            for index in np.random.permutation(len(train_meta)):
                meta = train_meta[index]
                responsibility = responsibilities[index]
                self.weight -= 1./t * self.gradient(self.weight, responsibility, meta, raw_data[meta["slice"]], alphas)
                t += 1
        return t
        
    def vSGD_fd(self, responsibilities, train_meta, raw_data, alphas=(1e-5,1e-5), rho=1e-5, batch_size=1):
        """
        0: g : exp_gradient
        1: v : var_gradient
        2: h_fd: exp_curvature
        3: v_fd: var_curvature
        """
        dim = self.weight_dim()
        offset = 1e-5
        
        moving_avgs = np.zeros((4,dim))
        window_sizes = np.ones(dim)
        
        iteration = 0
        rate = 1.0
        epoch = len(train_meta)
        while iteration < epoch or (iteration < 5*epoch and rho < rate):  # At least 1 epoch
            # Sample batch
            batch_indexes = np.random.randint(epoch, size=batch_size)
            
            # Compute gradient and shifted gradient for each sample in the batch
            # Compute approx. curvature and update cumulative stats
            cumul = np.zeros((4,dim))
            for index in batch_indexes:                                
                responsibility = responsibilities[index]
                meta = train_meta[index]
    
                gradient = self.gradient(self.weight, responsibility, meta, raw_data[meta["slice"]], alphas)
                curvature = self.gradient(self.weight + moving_avgs[0], responsibility, meta, raw_data[meta["slice"]], alphas)
                curvature = abs((gradient-curvature)/(moving_avgs[0]+offset))
                
                cumul[0] += gradient
                cumul[1] += gradient**2
                cumul[2] += curvature
                cumul[3] += curvature**2
            cumul /= batch_size
            

            # Numpy/ vectorized version
            outlier = abs(cumul[0]-moving_avgs[0]) > 2 * np.sqrt(moving_avgs[1]-moving_avgs[0]**2)
            outlier = outlier + abs(cumul[2]-moving_avgs[2]) > 2*np.sqrt(moving_avgs[3]-moving_avgs[2]**2)
            window_sizes[outlier] += 1 
            
            # Update moving averages
            moving_avgs += (cumul - moving_avgs) / (window_sizes+offset)
            
            # Update memory size
            window_sizes += 1 - window_sizes * moving_avgs[0]**2 / (moving_avgs[1]+offset)
                
            if iteration > 100:          # enable bootstrapping of moving windows
                # Estimate learning rates
                temp = moving_avgs[0]**2
                etas = batch_size * temp / (offset + moving_avgs[1] + (batch_size-1) * temp)
                etas *= moving_avgs[2] / (moving_avgs[3]+offset)
                
                # Update weight
                self.weight -= etas * cumul[0]
                
                rate = etas.mean()
            iteration += 1
      
        return iteration

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

cdef class LogitBase(AbstractCRF):
    
    cdef public object w_unary
        
    def __init__(self, states, labels, dimension, mu=0., sigma=10.):
        
        self.s_encoder = LabelEncoder()
        self.s_encoder.fit(states)
        self.K = len(states)
        
        self.l_encoder = LabelEncoder()
        self.l_encoder.fit(labels)
        self.G = len(labels)
        
        self.boundary = dimension
        self.w_unary = np.random.normal(mu,sigma, 1+self.K*(1+self.G+dimension))
        
    def getWeight(self):
        return self.w_unary
    
    def setWeight(self, w):
        self.w_unary = w
        
    def weight_dim(self):
        return self.w_unary.size
        
    weight = property(getWeight, setWeight)
    
    def gradient(self, weight, responsibility, meta, raw_data, alpha):
        
        grad_unary = alpha * weight
        grad_unary[0] = 0.  # intercept is not regularized
        
        mu_old = responsibility
        T = meta["length"]
        if self.G == 1:
            glabel = self.l_encoder.transform([meta["glabel"]])[0]
        else:
            glabel = 0
        
        
        # Compute probabilities given new weight vector
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, raw_data)
        pair = np.ones((self.G, T, self.K, self.K))
        forward, backward, scale = self.forward_backward(unary.values, pair)
            
        # Class label probability
        Z_new = forward[:,-1].sum()
                
        # Unary quantities, given all observations
        joint_new = forward * backward / Z_new

        
        # Compute gradient
        grad_unary[0] -= 1.0/unary.scale * mu_old[glabel].sum()
        for s in range(self.K):
            offset = 1 + unary.shift * s
            select = slice(offset+1+self.G, offset+unary.shift)
            
            grad_unary[offset] -= np.dot(mu_old[glabel,:,s], (1.0+np.arange(T)) / (T*unary.scale))
            grad_unary[offset+1+glabel] -= mu_old[glabel,:,s].sum() / unary.scale
            grad_unary[select] -= (mu_old[glabel,:,s] * unary.observations) / unary.scale
            
            grad_unary += 1.0/unary.scale * joint_new.sum()
            for g in range(self.G):
                grad_unary[offset] += np.dot(joint_new[g,:,s], (1.0+np.arange(T)) / (T*unary.scale))
                grad_unary[offset+1+g] += joint_new[g,:,s].sum() / unary.scale
                grad_unary[select] += (joint_new[g,:,s] * unary.observations) / unary.scale
                
            
        return grad_unary
        
    def expectation_entry(self, entry):

        if entry[2] and entry[0]["labels"]!= []:
            return self.observed(entry[0]["glabel"], entry[0]["labels"], ["unary"])[0]
            
        else:
            unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
            pair = np.ones((self.G, entry[0]["length"], self.K, self.K))
        
            forward, backward, scale = self.forward_backward(unary.values, pair)
            
            # Class label probability
            Z = forward[:,-1].sum(axis=1)
                
            # Unary quantities, given all observations
            mu_unary = forward * backward
            for g in range(self.G):
                mu_unary[g] /= Z[g]
            
            return mu_unary

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

cdef class CRFBase(AbstractCRF):
    
    cdef:
        public object w_unary, w_pair
        public int sep
    
    def __init__(self, states, labels, dimensions, mu_unary=0., sigma_unary=10., mu_pair=0., sigma_pair=10.):
        
        self.s_encoder = LabelEncoder()
        self.s_encoder.fit(states)
        self.K = len(states)
        
        self.l_encoder = LabelEncoder()
        self.l_encoder.fit(labels)
        self.G = len(labels)
        
        self.boundary = dimensions[0]

        self.w_unary = np.random.normal(mu_unary,sigma_unary, 1+self.K*(1+self.G+dimensions[0]))
        self.w_pair = np.random.normal(mu_pair,sigma_pair, 1+self.K*(1 + self.G*self.K + dimensions[1]))
        self.sep = self.w_unary.size
    
    def getWeight(self):
        return np.hstack([self.w_unary, self.w_pair])
    
    def setWeight(self, w):
        self.w_unary = w[:self.sep]
        self.w_pair = w[self.sep:]
        
    def weight_dim(self):
        return self.w_unary.size + self.w_pair.size
        
    weight = property(getWeight, setWeight)
    
    def gradient(self, weight, responsibility, meta, raw_data, alphas):
            
        mu_unary_old, mu_pair_old = responsibility
        T = meta["length"]
        if self.G == 1:
            glabel = 0
        else:
            glabel = self.l_encoder.transform([meta["glabel"]])[0]
            
        # Priors
        grad_unary = alphas[0] * weight[:self.sep]
        grad_unary[0] = 0.
        grad_pair = alphas[1] * weight[self.sep:]
        grad_pair[0] = 0.
        
        # Compute probabilities given new weight vector
        unary = UnaryFactor(self.K, self.G, weight[:self.sep], self.boundary, raw_data)
        pair = PairFactor(self.K, self.G, weight[:self.sep], self.boundary, raw_data)
    
        forward, backward, scale = self.forward_backward(unary.values, pair.values)
        
        # Class label probability
        Z_new = forward[:,-1].sum()
            
        # Unary quantities, given all observations
        joint_unary_new = forward * backward / Z_new
        
        # Pair quantities, given all observations
        joint_pair_new = np.empty((self.G, T, self.K, self.K))
        for t in range(1, T):
            for s1 in range(self.K):
                for s2 in range(self.K):
                    joint_pair_new[:, t, s1, s2] = forward[:, t-1, s1] * unary.values[:, t, s2] 
                    joint_pair_new[:, t, s1, s2] *= pair.values[:, t, s1, s2] * backward[:, t, s2]  / (Z_new * scale[t])
            
        
        # Compute gradient
        # Unary
        grad_unary[0] -= 1.0/unary.scale * mu_unary_old[glabel].sum()
        for s in range(self.K):
            offset = 1 + unary.shift * s
            select = slice(offset+1+self.G, offset+unary.shift)
            
            grad_unary[offset] -= np.dot(mu_unary_old[glabel,:,s], (1.0+np.arange(T)) / (T*unary.scale))
            grad_unary[offset+1+glabel] -= mu_unary_old[glabel,:,s].sum() / unary.scale
            grad_unary[select] -= (mu_unary_old[glabel,:,s] * unary.observations) / unary.scale
            
            
            grad_unary += 1.0/unary.scale * joint_unary_new.sum()
            for g in range(self.G):
                grad_unary[offset] += np.dot(joint_unary_new[g,:,s], (1.0+np.arange(T)) / (T*unary.scale))
                grad_unary[offset+1+g] += joint_unary_new[g,:,s].sum() / unary.scale
                grad_unary[select] += (joint_unary_new[g,:,s] * unary.observations) / unary.scale
            
        # Pair
        for t in range(1,T):
            for s1 in range(self.K):
                for s2 in range(self.K):
                    grad_pair -= mu_pair_old[glabel,t,s1,s2] * pair.featureVector(glabel,t,s1,s2)
                    
                    for g in range(self.G):
                        grad_pair += joint_pair_new[g,t,s1,s2] * pair.featureVector(g,t,s1,s2)
    
        return np.hstack([grad_unary, grad_pair])

    def expectation_entry(self, entry):

        if entry[2] and entry[0]["labels"]!= []:
            mu_unary, mu_pair = self.observed(entry[0]["glabel"], entry[0]["labels"], ["unary", "pair"])
            
            return mu_unary, mu_pair
            
        else:
            unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
            pair = PairFactor(self.K, self.G, self.w_pair, self.boundary, entry[1])
        
            forward, backward, scale = self.forward_backward(unary.values, pair.values)
            
            # Class label probability
            Z = forward[:,-1].sum(axis=1)
                
            # Unary quantities, given all observations
            mu_unary = forward * backward
            for g in range(self.G):
                mu_unary[g] /= Z[g]
            
            # Pair quantities, given all observations
            mu_pair = np.empty((self.G, entry[0]["length"], self.K, self.K))
            for g in range(self.G):
                for t in range(1, entry[0]["length"]):
                    for s1 in range(self.K):
                        for s2 in range(self.K):
                            mu_pair[g, t, s1, s2] = forward[g, t-1, s1] * unary.values[g,t, s2] 
                            mu_pair[g, t, s1, s2] *= pair.values[g, t, s1, s2] * backward[g, t, s2]  / (Z[g] * scale[t])
            
            return mu_unary, mu_pair

    def training_report(self, train_meta, train_data, test_meta, test_data, alphas, start, EMiteration, SGDiteration):
    
        summary = super(CRFBase, self).training_report(train_meta, train_data, test_meta, test_data, alphas, start, EMiteration, SGDiteration)
        summary["norm_w_pair"] = abs(self.w_pair**2).sum()
        
        return summary

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

cdef class Logit(LogitBase):
    
    def __init__(self, states, dimension, mu=0., sigma=10.):
        super(Logit, self).__init__(states, [0], dimension, mu, sigma)
      
    def predict_proba_entry(self, entry):
        
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair = np.ones((self.G, entry[0]["length"], self.K, self.K))

        forward, backward, _ = self.forward_backward(unary.values, pair)
        return (forward * backward)[0]
         
    def predict_entry(self, entry):
        
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair = np.zeros((self.G, entry[0]["length"], self.K, self.K))
        return self.s_encoder.inverse_transform(self.viterbi(unary.log, pair)[0])

    def test_error(self, test_meta, test_data):
        
        sent_confusion_matrix = np.zeros((self.K,self.K), dtype=float)
        for entry in test_meta:
            unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, test_data[entry["slice"]])
            pair_log = np.zeros((self.G, entry["length"], self.K, self.K))
            
            pred = self.viterbi(unary.log, pair_log)[0][0]
            true = self.observed(0, entry["labels"], ["unary"])[0][0]
            
            sent_confusion_matrix += np.dot(pred.T, true)
        
        # Compute rec. prec. F1, acc.
        return (self.toScores(sent_confusion_matrix, self.s_encoder.classes_),)

    def neg_log_likelihood(self, train_meta, train_data, alphas):
        
        alpha = alphas[0]
        neg_log_like = 0.5 * alpha * (self.w_unary[1:]**2).sum()
        gen = (meta for meta in train_meta if len(meta["labels"]) != 0)
        for meta in gen:
            proba = self.predict_proba_entry((meta, train_data[meta["slice"]]))
            states = self.s_encoder.transform(meta["labels"])
            for t, s in enumerate(states):
                neg_log_like -= np.log(proba[t,s])
        return neg_log_like

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

cdef class Topic(LogitBase):
                   
    def predict_proba_entry(self, entry, types=["joint", "seg", "class"]):
        
        answer = {}
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair = np.ones((self.G, entry[0]["length"], self.K, self.K))
        
        forward, backward, _ = self.forward_backward(unary.values, pair)

        mu_joint = forward * backward
        Z = forward[:,-1].sum(axis=1)
        mu_joint = forward * backward /  Z.sum()
        
        if "joint" in types:
            answer["joint"] = mu_joint
        
        if "seg" in types:
            answer["seg"] = mu_joint.sum(axis=0) 
            
        if "class" in types:
            answer["class"] = Z/Z.sum()
            
        return answer
        
    def predict_entry(self, entry, types=["joint", "seg", "class"]):
        
        answer = {}
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair_values = np.ones((self.G, entry[0]["length"], self.K, self.K), dtype=float)
        pair_log = np.zeros((self.G, entry[0]["length"], self.K, self.K), dtype=float)
        forward, backward, _ = self.forward_backward(unary.values, pair_values)

        Z = forward[:,-1].sum(axis=1)

        if "joint" in types:
            paths, proba = self.viterbi(unary.log, pair_log)
            proba = np.exp(proba) / Z.sum()
            best_class = proba.argmax()
            best_seq = paths[best_class].argmax(axis=1)
            answer["joint"] = { "class":self.l_encoder.inverse_transform([best_class])[0],\
                                "seq":self.s_encoder.inverse_transform(best_seq)}

        if "seg" in types:
            temp = self.viterbi_segmentation(unary.log.sum(axis=0), pair_log.sum(axis=0))[0]
            labels = temp.argmax(axis=1)
            answer["seg"] = self.s_encoder.inverse_transform(labels)
            
        if "class" in types:
            best_class = Z.argmax()
            answer["class"] = self.l_encoder.inverse_transform([best_class])[0]
            
        return answer
        
    def test_error(self, test_meta, raw_data):
        
        document_confusion_matrix = np.zeros((self.G,self.G), dtype=float)
        sent_confusion_matrix = np.zeros((self.K,self.K), dtype=float)
        
        for entry in test_meta:
            # Utility values
            unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, raw_data[entry["slice"]])
            pair_values = np.ones((self.G, entry["length"], self.K, self.K))
            pair_log = np.zeros((self.G, entry["length"], self.K, self.K))
            
            
            # Compute model prediction
            forward, backward, scale = self.forward_backward(unary.values, pair_values)
            Z = forward[:, -1].sum(axis=1)
            
            
            pred = self.viterbi_segmentation(unary.log.sum(axis=0), pair_log.sum(axis=0))[0]
            best_class = Z.argmax()
            
            # True values
            glabel = self.l_encoder.transform([entry["glabel"]])[0]
            true = self.observed(entry["glabel"], entry["labels"], ["unary"])[0][glabel]
            
            # Update confusion matrices
            sent_confusion_matrix += np.dot(pred.T, true)
            document_confusion_matrix[best_class, glabel] += 1
        
        # Compute rec. prec. F1, acc.
        sent_scores = self.toScores(sent_confusion_matrix, self.s_encoder.classes_)
        doc_scores = self.toScores(document_confusion_matrix, self.l_encoder.classes_)
        return (sent_scores, doc_scores)
        
    def neg_log_likelihood(self, train_meta, raw_data, alphas):
        alpha = alphas[0]
        neg_ll = 0.5 * alpha * (self.w_unary[1:]**2).sum()
        for meta in train_meta:
            proba = self.predict_proba_entry((meta, raw_data[meta["slice"]]), ["class"])["class"]
            glabel = self.l_encoder.transform([meta["glabel"]])[0]
            neg_ll -= np.log(1e-5+proba[glabel])
            
        return neg_ll

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

cdef class LinearChainCRF(CRFBase):
    
    def __init__(self, states, dimensions, mu_unary=0., sigma_unary=10., mu_pair=0., sigma_pair=10.):
        
        super(LinearChainCRF, self).__init__(states, [0], dimensions, mu_unary, sigma_unary, mu_pair, sigma_pair)

    def predict_proba_entry(self, entry):

        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair = PairFactor(self.K, self.G, self.w_pair, self.boundary, entry[1])
        
        forward, backward, _ = self.forward_backward(unary.values, pair.values)
        return (forward * backward)[0]
        
    def predict_entry(self, entry):
        
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair = PairFactor(self.K, self.G, self.w_pair, self.boundary, entry[1])
        return self.s_encoder.inverse_transform(self.viterbi(unary.log, pair.log)[0])
                
    def test_error(self, test_meta, raw_data):
        
        sent_confusion_matrix = np.zeros((self.K,self.K), dtype=float)
        
        for entry in test_meta:
            unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, raw_data[entry["slice"]])
            pair = PairFactor(self.K, self.G, self.w_pair, self.boundary, raw_data[entry["slice"]])
            
            pred = self.viterbi(unary.log, pair.log)[0][0]
            true = self.observed(0, entry["labels"], ["unary"])[0][0]

            sent_confusion_matrix += np.dot(pred.T, true)
        
        # Compute rec. prec. F1, acc.
        return (self.toScores(sent_confusion_matrix, self.s_encoder.classes_),)

    def neg_log_likelihood(self, train_meta, raw_data, alphas):
        
        neg_ll = 0.5 * (alphas[0] *(self.w_unary[1:]**2).sum() + alphas[1] * (self.w_pair[1:]**2).sum() )
        gen = (meta for meta in train_meta if len(meta["labels"])!=0)
        for meta in gen:
            proba = self.predict_proba_entry((meta, raw_data[meta["slice"]]))
            
            states = self.s_encoder.transform(meta["labels"])
            for t, s in enumerate(states):
                neg_ll -= np.log(proba[t,s])
        return neg_ll

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

cdef class HCRF(CRFBase):

    def predict_proba_entry(self, entry, types=["joint", "seg", "class"]):
        
        answer = {}
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair = PairFactor(self.K, self.G, self.w_pair, self.boundary, entry[1])
        forward, backward, _ = self.forward_backward(unary.values, pair.values)
        
        Z = forward[:,-1, :].sum(axis=1)
        mu_unary = forward * backward / Z.sum()
        
        if "joint" in types:
            answer["joint"] = mu_unary
        
        if "seg" in types:
            answer["seg"] = mu_unary.sum(axis=0) 
            
        if "class" in types:
            answer["class"] = Z/Z.sum()
            
        return answer
        
    def predict_entry(self, entry, types=["joint", "seg", "class"]):

        answer = {}
        unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, entry[1])
        pair = PairFactor(self.K, self.G, self.w_pair, self.boundary, entry[1])
        forward, backward, scale = self.forward_backward(unary.values, pair.values)

        Z = forward[:,-1].sum(axis=1)

        if "joint" in types:
            paths, proba = self.viterbi(unary.log, pair.log)
            proba = np.exp(proba) / Z.sum()
            best_class = proba.argmax()
            best_seq = paths[best_class].argmax(axis=1)
            answer["joint"] = { "class":self.l_encoder.inverse_transform([best_class])[0],\
                                "seq":self.s_encoder.inverse_transform(best_seq)}

        if "seg" in types:
            temp = self.viterbi_segmentation(unary.log.sum(axis=0), pair.log.sum(axis=0))[0]
            labels = temp.argmax(axis=1)
            answer["seg"] = self.s_encoder.inverse_transform(labels)
            
        if "class" in types:
            best_class = Z.argmax()
            answer["class"] = self.l_encoder.inverse_transform([best_class])[0]
            
        return answer

    def test_error(self, test_meta, raw_data):
        
        document_confusion_matrix = np.zeros((self.G,self.G), dtype=float)
        sent_confusion_matrix = np.zeros((self.K,self.K), dtype=float)
        
        for entry in test_meta:
            # Compute model prediction
            unary = UnaryFactor(self.K, self.G, self.w_unary, self.boundary, raw_data[entry["slice"]])
            pair = PairFactor(self.K, self.G, self.w_pair, self.boundary, raw_data[entry["slice"]])
            forward, backward, scale = self.forward_backward(unary.values, pair.values)
            Z = forward[:, -1].sum(axis=1)
            
        
            pred = self.viterbi_segmentation(unary.log.sum(axis=0), pair.log.sum(axis=0))[0]
            best_class = Z.argmax()
            
            # True values
            glabel = self.l_encoder.transform([entry["glabel"]])[0]
            true = self.observed(entry["glabel"], entry["labels"], ["unary"])[0][glabel]
            
            # Update confusion matrices
            sent_confusion_matrix += np.dot(pred.T, true)
            document_confusion_matrix[best_class, glabel] += 1
        
        # Compute rec. prec. F1, acc.
        sent_scores = self.toScores(sent_confusion_matrix, self.s_encoder.classes_)
        doc_scores = self.toScores(document_confusion_matrix, self.l_encoder.classes_)
        return (sent_scores, doc_scores)
        
    def neg_log_likelihood(self, train_meta, raw_data, alphas):
        
        neg_ll = 0.5 * (alphas[0] * (self.w_unary[1:]**2).sum() + alphas[1] * (self.w_pair[1:]**2).sum() )
        for meta in train_meta:
            glabel = self.l_encoder.transform([meta["glabel"]])[0]
            proba = self.predict_proba_entry((meta, raw_data[meta["slice"]]), ["class"])["class"]
            neg_ll -= np.log(1e-5+proba[glabel])
            
        return neg_ll
