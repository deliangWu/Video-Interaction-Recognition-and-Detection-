import multiprocessing
import numpy as np
import tensorflow as tf
import time
import sys
import ucf101
import videoPreProcess as vpp


class A:
    def __init__(self,din):
        self._a = din + 10
        self._b = din + 20
    
    def getB(self):
        return self._b

if __name__ == "__main__":
    ta = A(3)
    print(ta._a)
    print(A(3).getB())
    
    