import numpy as np
import RAMP as model

#-------------------------------------------------------
#  helper functions
#-------------------------------------------------------

def extract_time_series(filename):
    """
    Extracts the data from file, for a scientific workflow
    #-------------------------
    @param filename : the filename
    @return [time,event,ip] multivariete times series
    """
    time_ = []
    event_ = []
    ip_ = []
    f = open(filename,"r")
    for line in f:
        entry = line.split()
        time_.append(int(entry[0]))
        event_.append(int(entry[1]))
        ip_.append(int(entry[2]))
    #---------------------
    f.close()
    data = np.array([time_,event_,ip_])
    return data    

def extract_truth_labels(filename):
    truth_labels = []
    with open(filename,"r") as f:
        for line in f.readlines():
            line = line.split()
            truth_labels.append(line)
    return truth_labels

    
#-------------------------------------------------------
#  Run RAMP
#-------------------------------------------------------

subseq_length = 5
feedback_period = 200
num_features = 3
bias = 0.85
start_index = [0,516,767]
theta = 0.5
user_feedback = True
p_limit = 1

data = extract_time_series('data/sciflow_data.txt')
labels = extract_truth_labels('data/truth_labels.txt')



RAMP = model.RAMP(subseq_length,feedback_period,num_features,bias,start_index,theta,user_feedback,p_limit)
anomaly_flas, anomaly_scores, contrib = RAMP.run_RAMP(data,labels)

