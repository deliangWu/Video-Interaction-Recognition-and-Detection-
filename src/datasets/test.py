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
        gv,l = ucfset.loadTest(5)
        print(gv.shape)
        g = np.append(g,gv,1)
        print(g.shape)
        print("--------------sub-process-------------------------------")
        q.put(g)

if __name__ == "__main__":
    ucfset = ucf101.ucf101(frmSize=(112,128,3), numOfClasses=6)
    ucfset.loadTest(1)
    print('Load video ok')
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target = worker, args = (ucfset,q,))
    p.start()
    time.sleep(2)
    while True:
        vg = q.get_nowait()
        print("--------------father-process-------------------------------")
        print("The time for father process is {0}".format(time.ctime()))
        print(vg.shape)
        print("--------------father-process-------------------------------")
        print(' ')
        time.sleep(2)
    p.join()