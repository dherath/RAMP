import numpy as np
import matplotlib.pyplot as plt

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
            line = line.strip()
            truth_labels.append(int(line))
    return truth_labels

    
#-------------------------------------------------------
#  Run RAMP
#-------------------------------------------------------

subseq_length = 5
feedback_period = 60
num_features = 3
bias = 0.85
start_index = [0]
theta = 0.2
user_feedback = True
p_limit = 1

data = extract_time_series('data/toy_dataset.txt')
labels = extract_truth_labels('data/toy_truth_labels.txt')

#print(labels)

RAMP = model.RAMP(subseq_length,feedback_period,num_features,bias,start_index,theta,user_feedback,p_limit)
anomaly_flags, anomaly_scores, contrib = RAMP.execute(data,labels)

# plotting results
print("plotting")
thresh = np.ones([np.size(anomaly_scores)])*theta
plt.figure()
plt.subplot(211)
plt.plot(anomaly_scores)
plt.plot(thresh)
plt.ylim((0,1))


plt.subplot(212)
plt.plot(labels)

#plt.subplot(313)
#plt.plot(workflow)

plt.show()
