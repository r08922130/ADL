import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import math
arg = sys.argv

reward1 = arg[1]
#reward2 = arg[2]
c_dict = {}
x_dict = {}
plt.figure()
min_value = 999
max_value = -999
window_size = 100
index = -5
step_index = -9
step = 50
argc = len(arg)-1 
color = ['r-','g-','b-','m-']
if 'pg' in reward1:
    index = -2
    step_index = -6
    step = 10
    window_size = 10


for style, name in zip(color[:argc],arg[1:]):
    if name == "N":
        break
    with open(name+'_log.txt','r') as f:
        lines = f.readlines()
        c_dict[name] = []
        x_dict[name] = []
        for line in lines:
            #print(line.split(' ')[1])
            sp = line.split(' ')
            r = float(sp[index])
            steps = int(sp[step_index].split('/')[0])
            c_dict[name] += [r]
            if 'pg' not in reward1:
                x_dict[name] += [steps]
            
        
        cumsum, moving_aves = [0], []

        for i, x in enumerate(c_dict[name], 1):
            cumsum.append(cumsum[i-1] + x)
            if i>=window_size:
                moving_ave = (cumsum[i] - cumsum[i-window_size])/window_size
                #can do stuff with moving_ave here
                min_value = min(moving_ave,min_value)
                max_value = max(moving_ave,max_value)
                moving_aves.append(moving_ave)
        if 'pg' in reward1:
            plt.plot(moving_aves,style, label=name.split('log_')[-1])
        else:
            
            plt.plot(x_dict[name][window_size-1:],moving_aves,style, label=name.split('log_')[-1])
if 'pg' not in reward1:
    plt.xticks(np.arange(0, 3000000, 200000))
print(max_value)
step = (max_value - min_value)//10
plt.yticks(np.arange(min_value//1, max_value//1 +step, step))
plt.legend()
plt.savefig('output.jpg')