from datetime import datetime
import numpy as np
def Wdatetime(fileName,flag):
    f = open(fileName,'a+')
    if flag == 'start':
        string = 'The start time is ' + str(datetime.now()) + '\n'
    elif flag == 'end':
        string = 'The end time is ' + str(datetime.now()) + '\n'
    else:
        string = str(datetime.now())
    f.write(string)
    f.close()
    return None