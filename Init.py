__author__ = 'MatthiasFuchs'

import pandas as pd
import numpy as np


path = '/Users/MatthiasFuchs/Desktop/Daten+Labels/NWDaten/ID001/20150901/merged.csv'
rowData = pd.read_csv(path, sep= '\t')
data = rowData.as_matrix()

#sensors: 'RNS', 'RLA', 'RUA', 'STE', 'LUA', 'LLA', 'LNS', 'RUF', 'RLL', 'RUL', 'CEN', 'LUL', 'LLL', 'LUF'
#datas: 'acc', 'gyr', 'mag', 'qua'
#time: True, False
#return: Matrix mit ausgewaehlten Werten

def getData(sensors, datas, time = True):


    sensorBools = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    if (time == True):
        output = data[:, 0:2]
    else:
        output = np.empty(shape=(len(data[:, 0]), 0))

    start, stop = startStop(datas)
    defaultOffset = 2
    for sensor in sensors:
        if (sensor == 'RNS' and sensorBools[0] == False):
            print 'RNS'
            offset = 0
            sensorBools[0] = True
        elif (sensor == 'RLA' and sensorBools[1] == False):
            print 'RLA'
            offset = 1
            sensorBools[1] = True
        elif (sensor == 'RUA' and sensorBools[2] == False):
            print 'RUA'
            offset = 2
            sensorBools[2] = True
        elif (sensor == 'STE' and sensorBools[3] == False):
            print 'STE'
            offset = 3
            sensorBools[3] = True
        elif (sensor == 'LUA' and sensorBools[4] == False):
            print 'LUA'
            offset = 4
            sensorBools[4] = True
        elif (sensor == 'LLA' and sensorBools[5] == False):
            print 'LLA'
            offset = 5
            sensorBools[5] = True
        elif (sensor == 'LNS' and sensorBools[6] == False):
            print 'LNS'
            offset = 6
            sensorBools[6] = True
        elif (sensor == 'RUF' and sensorBools[7] == False):
            print 'RUF'
            offset = 7
            sensorBools[7] = True
        elif (sensor == 'RLL' and sensorBools[8] == False):
            print 'RLL'
            offset = 8
            sensorBools[8] = True
        elif (sensor == 'RUL' and sensorBools[9] == False):
            print 'RUL'
            offset = 9
            sensorBools[9] = True
        elif (sensor == 'CEN' and sensorBools[10] == False):
            print 'CEN'
            offset = 10
            sensorBools[10] = True
        elif (sensor == 'LUL' and sensorBools[11] == False):
            print 'LUL'
            offset = 11
            sensorBools[11] = True
        elif (sensor == 'LLL' and sensorBools[12] == False):
            print 'LLL'
            offset = 12
            sensorBools[12] = True
        elif (sensor == 'LUF' and sensorBools[13] == False):
            print 'LUF'
            offset = 13
            sensorBools[13] = True
        else:
            offset = -1
            print 'Keine gueltige Eingabe oder doppelte Eingabe (sensors)'

        if (offset != -1):
            for i in range(0, len(start)):
                if(start[i] != -1 and stop[i] != -1):
                    sensorOffset = 13*offset
                    startData = defaultOffset+sensorOffset+start[i]
                    stopData = defaultOffset+sensorOffset+stop[i]
                    output = np.c_[output, data[:, startData:stopData]]
                    print ("start: %d stop: %d" % (startData, stopData))
    np.savetxt('selectedData.csv', output, delimiter='\t')

    return output


def startStop(datas):
    qua = False
    acc = False
    gyr = False
    mag = False

    startOffsets = np.zeros(len(datas))
    stopOffsets = np.zeros(len(datas))

    index = 0
    for data in datas:
        if (data == 'acc' and acc == False):
            acc = True
            startOffsets[index] = 0
            stopOffsets[index] = 3
        elif (data == 'gyr' and gyr == False):
            gyr = True
            startOffsets[index] = 3
            stopOffsets[index] = 3
        elif (data == 'mag' and mag == False):
            mag = True
            startOffsets[index] = 6
            stopOffsets[index] = 9
        elif (data == 'qua' and qua == False):
            qua = True
            startOffsets[index] = 9
            stopOffsets[index] = 13
        else:
            startOffsets[index] = -1
            stopOffsets[index] = -1
            print 'Keine gueltige Eingabe oder doppelte Eingabe (datas)'
        index += 1

    return startOffsets, stopOffsets

output = getData(['RNS', 'LUF'], ['acc', 'qua'], False)


