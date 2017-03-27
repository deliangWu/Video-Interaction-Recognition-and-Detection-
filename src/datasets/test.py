import multiprocessing
import time
import sys
import ucf101

def worker(ucfset):
    n = 5
    for i in range(10):
        print("The time is {0}".format(time.ctime()))
        ucfset.loadTest(1)

if __name__ == "__main__":
    ucfset = ucf101.ucf101(frmSize=(112,128,3), numOfClasses=6)
    p = multiprocessing.Process(target = worker, args = (ucfset,))
    ucfset.loadTest(1)
    print('Load video ok')
    p.start()
    while True:
        print("The time is {0}".format(time.ctime()))
        print("p.pid:", p.pid)
        print("p.name:", p.name)
        print("p.is_alive:", p.is_alive())    
        time.sleep(5)
    p.join()