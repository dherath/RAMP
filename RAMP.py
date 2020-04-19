import numpy as np

#--------------------------------------------------------------
#       RAMP : Real-Time Aggregated Matrix Profile
#--------------------------------------------------------------

class RAMP:
    """
    - Author: J. Dinal Herath
    - For details refer to paper: 
         'RAMP: Real-Time Anomaly Detection in Scientific Workflows' 
          by J. Dinal Herath, Changxin Bai, Guanhua Yan, Ping Yang, Shiyong Lu. 
          In: IEEE International Conference on Big Data (2019). 
    - Link:
         http://www.dinalherath.com/papers/2019RAMP_extended_paper.pdf
    """

    def __init__(self,subseq_length,feedback_period,num_features,bias,start_index,threshold,give_user_feedback,p_limit=1):
        """
        init function
        ---------------------
        @param subseq_lengtn : the length of a sub-sequence
        @param feedback_period : the length of T' used as comparison
        @param num_features : the dimensions of the time series
        @param start_index : defaults to [0] if it is a non-interleaved process, otherwise must have indices for when processes start
        @param threshold : theta, if anomaly score > theta then Anomaly == True, else False
        @param gives_user_feedback : defaults to True, as RAMP was intended with user involvement, but can be False
        @param p_limit : defaults to 1, if user feedback is given, an additional modification for reducing penalization for repititive anomalies
        """
        self.m = subseq_length
        self.M = feedback_period
        self.d = num_features
        self.b = bias
        self.start_index = start_index
        self.theta = threshold
        self.user_feedback = give_user_feedback
        self.p_limit = p_limit
        #--------------------
        self.num_proc = len(start_index)
        self.R = np.zeros([self.num_proc,self.d,self.M]) # the record relative distances for past M steps
        self.prob_previous = np.zeros([self.num_proc]) # the previous probability value per proc
        self.T_ = np.zeros([self.num_proc,self.d,self.M]) # previously S_, the comparison set
        self.K = np.zeros([self.num_proc],int) # parameter for uncertainty f()
        self.H = np.zeros([self.num_proc,self.M]) # record for ratio between (un weighted beta/theta) for M steps
        self.W = [dict() for i in range(len(start_index))] # dictionary of weight per process
        self.alpha = self.normalized_gaussian(0,self.m) # used for the training algorithm
        return

    #------------------------------------
    # Main Functions
    #------------------------------------

    def anomaly_detection(self,t,T,proc_id=0):
        """
        identifies if an anomaly is present in sub-sequence T
        -----------------------
        @param t : the time step [0,...,M-1] within the window M
        @param T : a multi-dimensional subsequence [d,m]
        @param proc_id : the id of the process if experiment is interleaved, 0 is default
        -----------------------
        @return [anomaly_detected,beta,C_t]
        """
        beta = 0 # anomaly score
        key = 0 # index of the weight dictionary
        C_t = np.zeros(self.d) # contribution to beta from all features/dimensions
        D_min = np.zeros(self.d) # an empty list to store min(Distance-value) for each d
        # computing the relative distance for each dimension/feature
        for j in range(self.d):
            min_rd = np.inf # initially has infinite relative distance
            min_k = -1 # a temp variable for computing the index of the weight 
            for k in range(self.M - self.m):
                abs_distance = np.sum(np.abs(np.subtract(T[j],self.T_[proc_id,j,k:k+self.m])))
                compared_with = np.sum(np.abs(self.T_[proc_id,j,k:k+self.m]))
                relative_distance = abs_distance / compared_with
                if relative_distance <  min_rd:
                    min_rd = relative_distance
                    min_k = k
            self.R[proc_id,j,t] = min_k
            key += ((self.M - self.m)**j)*min_k
            beta += min_rd
            D_min[j] = min_rd
        #--------------------
        C_t = (np.divide(D_min,beta) if beta > 0 else np.zeros(self.d))# computing the contribution to the anomaly
        self.H[proc_id,t] = (beta / self.theta if beta > 0 else 1e-10) # updating the ratio between (un weighted beta/theta)
        # computing the weighted score
        if int(key) in self.W[proc_id]:
            beta = self.W[proc_id][int(key)] * beta
        # get anomaly flag
        if beta > self.theta:
            return [True,beta,C_t]
        else:
            return [False,beta,C_t]

    def uncertainty_function(self,proc_time,proc_id=0):
        """
        gives a probabilistic value for the uncertainty of whether an anomaly flagged is TP or FP
        -----------------------
        @param proc_time : the time with respect to the individual process [0,...,inf]
        @param proc_id : the ID for the interleaved process, default = 0
        -----------------------
        @return probabilistic value [0,1], if ~1 then most likely TP
        """
        if proc_time > self.M:
            return np.abs(1 - np.exp(self.K[proc_id]**self.b - proc_time**self.b))
        else:
            return 0
        
    def adaptive_training(self,t,p,proc_id=0):
        """
        conducts one training cycle, does not return because W,H are defined in class
        -----------------------
        @param t: time step [0,...,M-1] within the range M
        @param p: the result from the uncertainty function
        @param proc_id : the process id, default = 0 for only one process
        """
        keys = np.zeros(2*self.m+1) # the indices of the weights to train
        for k in range(2*self.m+1):
            # compute the index for the weights
            for j in range(self.d):
                keys[k] += ((self.M-self.m)**j)*(self.R[proc_id,j,t] - self.m + k)
            # add weight into W, if it does not exist in W
            if keys[k] not in self.W[proc_id]:
                self.W[proc_id][keys[k]] = 1
            # weight updating step
            if k == self.m :
                beta_unweighted = (self.H[proc_id,t] * self.theta) / self.W[proc_id][keys[k]]
                self.W[proc_id][keys[k]]  *= ( (self.alpha[k]*(1-p)) / (2*self.H[proc_id,t]) )
                self.H[proc_id,t] = (self.W[proc_id][keys[k]] * beta_unweighted) / self.theta
            else:
                self.W[proc_id][keys[k]]  *= ( (self.alpha[k]*(1-p)) / (2*self.H[proc_id,t]) )
        return
    
    def human_in_the_loop_training(self,proc_time,num_fp,U_TP,proc_id=0):
        """
        optional periodic training, if user feedback is given, does not return
        because K,W are defined in class
        -----------------------
        @param proc_time : the time index for the current process [0,..,inf]
        @param num_fp : the total number of false positives marked
        @param U_TP : the indices for the True Positives [0,..,M-1]
        @param proc_id : the process id, default = 0 
        """
        # update step for the uncertainty function parameter
        self.K[proc_id] += (proc_time - self.K[proc_id])*(num_fp/self.M)
        # weight update step for True Positives
        for i in U_TP:
            key = 0 # the index of the weight
            for j in range(self.d):
                key += ((self.M-self.m)**j)*self.R[proc_id,j,i]
            if key not in self.W[proc_id]:
                self.W[proc_id][int(key)] = 1
            self.W[proc_id][int(key)] *= (2/self.H[proc_id,i])
        return
    
    def normalized_gaussian(self,mu,sigma):
        """
        defines a normalized gaussian values between [-sigma:sigma] and mu=0
        ------------------------
        @param mu : the mean (0)
        @param sigma : m (the subsequence length)
        ------------------------
        @return the normalized gaussian s with (2*m+1) samples
        """
        x = np.zeros(2*sigma+1)
        s = np.zeros(2*sigma+1)
        temp = -sigma
        for i in range(2*sigma+1):
            s[i] = (1/np.absolute(np.sqrt(2*np.pi*(sigma**2))))*(np.exp(-(temp-mu)**2)/(2*sigma**2))
            temp += 1
        s = s/max(s)
        return s
    
    #------------------------------------
    # Running RAMP
    #------------------------------------
    
    def execute(self,time_series,truth_labels=[]):
        """
        # an illustrative example of how to run RAMP
        # includes code steps to run a single non-interleaved process, proc_id == 0 is set as default
        # the steps in this function can be extended for interleaved processes as well
        # extending it will require handling additional a corner case for when a process is interleaved
        # for the first time while another process is already executing
        #------------------------
        @param time_series : the multivariate time series
        @param truth_labels : the truth labels, 1 = anomaly, 0 = benign (only needed if user_feedback == True)
        #------------------------
        @return [anomaly_flags,anomaly_scores,contributions]
        """
        # lists for the results
        beta_result = []
        C_result = []
        A_result = []
        #--------------------
        time = 0 # the time steps for the time series
        num_samples = np.size(time_series,1) - self.m # the total number of sub-sequences
        # Loop through all sub-sequences in time series 
        while time < num_samples:
            # num_fp and U_TP are only needed if user feedback is given
            num_fp = 0 # number of FP marked in window of M time steps
            U_TP = [] # true positive indices
            t = time  % self.M # identify the index of R,H w.r.t the current process execution
            # The first M  samples are considered as the set T' 
            if time < self.M :
                self.T_[0,:,:] = time_series[:,time:time+self.M] # the comparison set, firt M samples
                self.K[0] = self.M # p must be 0
                # fill in temporary values for beta, Anomaly flag, Contribution
                for i in range(self.M):
                    beta_result.append(-1)
                    A_result.append(0) # 0 = False
                    C_result.append([-1 for i in range(self.d)])
                # increment the proc_time by M steps
                time = self.M
            else:
                # For the rest of the time steps
                T = time_series[:,time:time+self.m] # the input subsequence
                # 1. call anomaly_detection() for the process
                anomaly_flag, beta, C_t = self.anomaly_detection(t,T)
                # append the results
                beta_result.append(beta)
                A_result.append(1 if anomaly_flag == True else 0)
                C_result.append(C_t)
                # 2. get uncertainty
                p = self.uncertainty_function(time)
                # 3. call adaptive_training()
                # modification: training will be done only if p < p_limit ~ 1
                # this will reduce the penalization for exactly repititive anomalies
                # when no user feedback is given
                # if user_feedback == True, then its fine to have p_limit = 1 (default)
                if p < self.p_limit:
                    self.adaptive_training(t,p)
                # -- optional human_in_the_loop_training() --
                # 4. update for user feedback if needed
                if self.user_feedback == True:
                    if anomaly_flag == True:
                        if truth_labels[time] == 1:
                            U_TP.append(t)
                        else:
                            num_fp += 1
                # 5. do human in the loop training ()
                if self.user_feedback == True:
                    self.human_in_the_loop_training(time,num_fp,U_TP)
                    # reset the number of false positives, TP indices from user feedback
                    num_fp = 0
                    U_TP = []
                # -- end of optional training --
                # 6. increment the time
                time += 1
        # --- end loop -------------------
        return A_result,beta_result,C_result
