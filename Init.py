__author__ = 'MatthiasFuchs'

import pandas as pd
import numpy as np




#Sucht die geforderten Daten im Datenset und gibt diese zurueck
#sensors: 'RNS', 'RLA', 'RUA', 'STE', 'LUA', 'LLA', 'LNS', 'RUF', 'RLL', 'RUL', 'CEN', 'LUL', 'LLL', 'LUF'
#datas: 'acc', 'gyr', 'mag', 'qua', 'rE1', 'rE2', 'rE3'
#inputData: Eingabedaten
#time: True, False
#return: Matrix mit ausgewaehlten Werten
def getData(inputData, sensors = ['RNS', 'RLA', 'RUA', 'STE', 'LUA', 'LLA', 'LNS', 'RUF', 'RLL', 'RUL', 'CEN', 'LUL', 'LLL', 'LUF'],
            datas = ['acc', 'gyr', 'mag', 'qua', 'rE1', 'rE2', 'rE3'], time = True, comment = ' '):

    sensorBools = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    if (time == True):
        output = inputData[:, 0:2]
    else:
        output = np.empty(shape=(len(inputData[:, 0]), 0))

    start, stop, dataString = startStop(datas)

    defaultOffset = 2
    sensorString = 'Sensoren: '

    for sensor in sensors:
        if (sensor == 'RNS' and sensorBools[0] == False):
            sensorString = sensorString + sensor + ' '
            print 'RNS'
            offset = 6
            sensorBools[0] = True
        elif (sensor == 'RLA' and sensorBools[1] == False):
            sensorString = sensorString + sensor + ' '
            print 'RLA'
            offset = 5
            sensorBools[1] = True
        elif (sensor == 'RUA' and sensorBools[2] == False):
            sensorString = sensorString + sensor + ' '
            print 'RUA'
            offset = 4
            sensorBools[2] = True
        elif (sensor == 'STE' and sensorBools[3] == False):
            sensorString = sensorString + sensor + ' '
            print 'STE'
            offset = 0
            sensorBools[3] = True
        elif (sensor == 'LUA' and sensorBools[4] == False):
            sensorString = sensorString + sensor + ' '
            print 'LUA'
            offset = 1
            sensorBools[4] = True
        elif (sensor == 'LLA' and sensorBools[5] == False):
            sensorString = sensorString + sensor + ' '
            print 'LLA'
            offset = 2
            sensorBools[5] = True
        elif (sensor == 'LNS' and sensorBools[6] == False):
            sensorString = sensorString + sensor + ' '
            print 'LNS'
            offset = 3
            sensorBools[6] = True
        elif (sensor == 'RUF' and sensorBools[7] == False):
            sensorString = sensorString + sensor + ' '
            print 'RUF'
            offset = 13
            sensorBools[7] = True
        elif (sensor == 'RLL' and sensorBools[8] == False):
            sensorString = sensorString + sensor + ' '
            print 'RLL'
            offset = 12
            sensorBools[8] = True
        elif (sensor == 'RUL' and sensorBools[9] == False):
            sensorString = sensorString + sensor + ' '
            print 'RUL'
            offset = 11
            sensorBools[9] = True
        elif (sensor == 'CEN' and sensorBools[10] == False):
            sensorString = sensorString + sensor + ' '
            print 'CEN'
            offset = 7
            sensorBools[10] = True
        elif (sensor == 'LUL' and sensorBools[11] == False):
            sensorString = sensorString + sensor + ' '
            print 'LUL'
            offset = 8
            sensorBools[11] = True
        elif (sensor == 'LLL' and sensorBools[12] == False):
            sensorString = sensorString + sensor + ' '
            print 'LLL'
            offset = 9
            sensorBools[12] = True
        elif (sensor == 'LUF' and sensorBools[13] == False):
            sensorString = sensorString + sensor + ' '
            print 'LUF'
            offset = 10
            sensorBools[13] = True
        else:
            offset = -1
            print 'Keine gueltige Eingabe oder doppelte Eingabe (sensors)'

        if (offset != -1):
            for i in range(0, len(start)):
                if(start[i] != -1 and stop[i] != -1):
                    sensorOffset = 22*offset
                    startData = defaultOffset+sensorOffset+start[i]
                    stopData = defaultOffset+sensorOffset+stop[i]
                    output = np.c_[output, inputData[:, startData:stopData]]
                    print ("start: %d stop: %d" % (startData, stopData))


    history = sensorString + '\n' + dataString + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    return output

#Sucht Start und Stopp in den Daten
def startStop(datas):
    qua = False
    acc = False
    gyr = False
    mag = False
    rE1 = False
    rE2 = False
    rE3 = False

    startOffsets = np.zeros(len(datas))
    stopOffsets = np.zeros(len(datas))

    string = 'Daten: '

    index = 0
    for data in datas:
        if (data == 'acc' and acc == False):
            string = string + data + ' '
            acc = True
            startOffsets[index] = 0
            stopOffsets[index] = 3
        elif (data == 'gyr' and gyr == False):
            string = string + data + ' '
            gyr = True
            startOffsets[index] = 3
            stopOffsets[index] = 6
        elif (data == 'mag' and mag == False):
            string = string + data + ' '
            mag = True
            startOffsets[index] = 6
            stopOffsets[index] = 9
        elif (data == 'qua' and qua == False):
            string = string + data + ' '
            qua = True
            startOffsets[index] = 9
            stopOffsets[index] = 13
        elif (data == 'rE1' and rE1 == False):
            string = string + data + ' '
            rE1 = True
            startOffsets[index] = 13
            stopOffsets[index] = 16
        elif (data == 'rE2' and rE2 == False):
            string = string + data + ' '
            rE2 = True
            startOffsets[index] = 16
            stopOffsets[index] = 19
        elif (data == 'rE3' and rE3 == False):
            string = string + data + ' '
            rE3 = True
            startOffsets[index] = 19
            stopOffsets[index] = 22
        else:
            startOffsets[index] = -1
            stopOffsets[index] = -1
            print 'Keine gueltige Eingabe oder doppelte Eingabe (datas)'
        index += 1

    return startOffsets, stopOffsets, string

