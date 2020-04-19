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

RAMP = model.RAMP(subseq_length,feedback_period,num_features,bias,start_index,theta,user_feedback,p_limit)
anomaly_flags, anomaly_scores, contrib = RAMP.execute(data,labels)


#-------------------------------------------------------
#  Plotting Results
#-------------------------------------------------------


thresh = np.ones(len(anomaly_scores))*theta
sz = len(anomaly_scores) # becuase the final data point will be at len(data) - self.m


plt.rcParams["font.size"] = "16"

plt.figure()

plt.subplot(311)
plt.plot(data[0,0:sz])
plt.title('time series dimension 1 (time-difference)')
plt.ylim((0,2000))

plt.subplot(312)
plt.plot(data[1,0:sz])
plt.title('time series dimension 2 (workflow-events)')

plt.subplot(313)
plt.plot(data[2,0:sz])
plt.title('time series dimension 3 (IP-change)')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.50)
plt.gcf().set_size_inches(18.5,8.5)
#plt.show()
plt.savefig('image_data.jpeg',farmeon=False,bbox_inches='tight')

plt.figure()

plt.subplot(311)
plt.plot(anomaly_scores,color = 'black',label = 'anomaly score')
plt.plot(thresh,color = 'red', label = 'threshold')
plt.title('RAMP - result (anomoalies)')
plt.ylim((0,1))
plt.legend(frameon = False,loc = 'upper left')

plt.subplot(312)
legends = ['dim 1 (time-difference)','dim 2 (workflow-events)', 'dim 3 (IP-change)']
for i in range(num_features):
    feature_contribution = []
    legend_label = legends[i]
    for value in contrib:
        feature_contribution.append(value[i])
    plt.plot(feature_contribution,label = legend_label)
plt.legend(frameon = False,loc = 'upper left')
plt.title('RAMP - result (contribution)')
plt.ylim((-0.1,1.1))

plt.subplot(313)
plt.plot(labels,color = 'black')
#plt.xlabel('Timestep',fontsize = '12')
plt.title('truth labels (1 - anomaly, 0 - benign)')


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.50)
plt.gcf().set_size_inches(18.5,8.5)
#plt.show()
plt.savefig('image_result.jpeg',farmeon=False,bbox_inches='tight')
