import multiprocessing
import numpy as np
import time
import sys
import ucf101
import videoPreProcess as vpp

def worker(ucfset,q):
    n = 5
    g = np.empty((3,1,16,112,128,3))
    q.put(g)
    for i in range(10):
        print("--------------sub-process-------------------------------")
        print("The time is {0}".format(time.ctime()))
        gv,l = ucfset.loadTest(2)
        print(gv.shape)
        g = np.append(g,gv,1)
        print(g.shape)
        print("--------------sub-process-------------------------------")
        q.put(g)

if __name__ == "__main__":
    print(int(sys.argv[1]))
    
    