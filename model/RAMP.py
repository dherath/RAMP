import numpy as np
import random
import sys

#--------------------------------------------------------------
#       RAMP : Real-Time Aggregated Matrix Profile
#--------------------------------------------------------------

class RAMP:

    def __init__(subseq_length,feedback_period,num_features,bias,start_index,threshold):
        """
        init function
        """
        self.m = subseq_length
        self.M = feedback_period
        self.d = num_features
        self.b = bias
        self.start_index = start_index
        self.theta = threshold
        #--------------------
        num_proc = len(start_index)
        self.R = np.zeros([num_proc,d,M]) # the record relative distances for past M steps
        self.prob_previous = np.zeros([num_proc]) # the previous probability value per proc
        self.T_ = np.zeros([num_proc,F,M]) # previously S_, the comparison set
        self.K = np.zeros([num_proc],int) # parameter for uncertainty f()
        self.H = np.zeros([num_proc,M]) # record for ratio between (beta/theta) for M steps
        self.W = [dict() for i in range(len(start_index))] # dictionary of weight per process
        
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
                abs_distance = np.sum(np.abs(np.subtract(T[j],T_[proc_id,j,k:k+self.m])))
                compared_with = np.sum(np.abs(self.T_[proc_id,j,k:k+self.m]))
                relative_distance = abs_distance / compared_with
                if relative_distance > min_rd:
                    min_rd = relative_distance
                    min_k = k
            self.R[proc_id,j,t] = min_k
            key += ((self.M - self.m)**j)*min_k
            beta += min_rd
            D_min[j] = min_rd
        #--------------------
        C_t = np.divide(D_min,beta)# computing the contribution to the anomaly
        self.H[proc_id,t] = beta / self.theta # updating the ratio between (beta/theta)
        # computing the weighted score
        if int(key) in W[proc_id]:
            beta = W[proc_id][int(key)] * beta
        # get anomaly flag
        if beta > self.theta:
            return [True,beta,C_t]
        else:
            return [False,beta,C_t]

    def uncertainty_function(self,proc_time,proc_id=0):
        """
        gives a probabilistic value for the uncertainty of TP/FP
        -----------------------
        @param proc_time : the tim with respect to the individual process
        @param proc_id : the ID for the interleaved process, default = 0
        -----------------------
        @return probabilistic value [0,1], if ~1 then most likely TP
        """
        return np.abs(1 - np.exp(self.K[proc_id]**b - proc_time**b))
        
    def adaptive_training(self,proc_id=0):
        """
        conducts one training cycle, does not return because W,H are defined in class
        """
        
        return
    
    def periodic_training(self,proc_id=0):
        """
        optional periodic training, if user feedback is given, does not return
        because K,W are defined in class
        """

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
    
    def run_RAMP(self,time_series,complete_user_feedback):
        self.N = self.normalized_gaussian(0,self.m)
        beta_result = []
        C_result = []
        A_result = []
        #--------------
        num_loaded_proc = 0
        all_procs_loaded = False # becomes True when all interleaved processes are loaded
        proc_index = 0 # the current index of the loaded process
        trial_period = False # First 2M instances
        time = 0 # the start time index
        time_period = 0 # an index value [0,M] to store inside records R,H
        num_samples = np.size(time_series,1) - self.m # the total number of sub-sequences
        user_feedback_record = np.ones([self.M])*(-1) # 0: if no anomaly, 1: is anomaly, -1: unmarked/shouldn't be used
        for time < num_samples:
            t = time_period % self.M
            user_feedback_record[t] = complete_user_feedback[time]
            # continue work ...
        return A_result,beta_result,C_result

    #------------------------------------
    # helper functions
    #------------------------------------

    def return_threshold(self):
        return self.theta

    def update_threshold(self,threshold):
        self.theta = threshold
        return
    
