from __future__ import print_function
import sys
import numpy as np

''' a method for print input string on terminal and write it to the file at the same time'''
def pAndWf(fileName, string):
    f = open(fileName,'a+')
    print(string,end='')
    f.write(string)
    f.close
    return None

def clearFile(fileName):
    f = open(fileName,'w')
    f.close()
    return None