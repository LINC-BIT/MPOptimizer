import torch
import torchvision
import torch.nn as nn
from re_train import retrain get_acc
K_parameter # List
t # micro-window number of seconds
T # window size of T sec
budget=T
def Aco(microbatch_size,checkpoint_size,freezing_rate):
    acc_i=get_acc()
	retrain(microbatch_size,checkpoint_size,freezing_rate,t)
    acc_f=get_acc()
    budget=budget-t
    return (acc_f-acc_i)/t
# return the accuracy gain
i=0
gain=dict()
for k in K_parameter:
    microbatch_size=k[0]
    checkpoint_size=k[1]
    freezing_rate=k[2]
    gain[i]=Aco(microbatch_size,checkpoint_size,freezing_rate)
    i=i+1

sorted(gain.items(),key=lambda x:-x[1])

while budget > t:
    for item,acc in gain.items():
        microbatch_size=K_parameter[item][0]
        checkpoint_size=K_parameter[item][1]
        freezing_rate=K_parameter[item][2]
        gain[item]=Aco(microbatch_size,checkpoint_size,freezing_rate)
        break
    sorted(gain.items(),key=lambda x:-x[1])
  
