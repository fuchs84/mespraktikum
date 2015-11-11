__author__ = 'MatthiasFuchs'

import pandas as pd
import numpy as np


defaultOffset = 2

#Sucht die geforderten Daten im Datenset und gibt diese zurueck
#sensors: 'STE', 'LUA', 'LLA', 'LNS', 'RUA', 'RLA', 'RNS', 'CEN', 'LUL', 'LLL', 'LUF', 'RUL', 'RLL', 'RUF'
#datas: 'acc', 'gyr', 'mag', 'qua', 'rE1', 'rE2', 'rE3'
#inputData: Eingabedaten
#time: True, False
#return: Matrix mit ausgewaehlten Werten
def getData(inputData, sensors = ['STE', 'LUA', 'LLA', 'LNS', 'RUA', 'RLA', 'RNS', 'CEN', 'LUL', 'LLL', 'LUF', 'RUL', 'RLL', 'RUF'],
            datas = ['acc', 'gyr', 'mag', 'qua', 'rE1', 'rE2', 'rE3'], specifiedDatas = ['w', 'x', 'y', 'z'], time = True):

    if (time == True):
        output = inputData[:, 0:2]
    else:
        output = np.empty(shape=(len(inputData[:, 0]), 0))

    dataRange, dataString = startStopData(datas, specifiedDatas)
    startSensors, sensorString = startSensor(sensors)


    for sensor in startSensors:
        if (sensor != -1):
            for data in dataRange:
                sensorOffset = 22*sensor+data+defaultOffset;
                output = np.c_[output, inputData[:, sensorOffset]]

    history = 'getData: \n'  + sensorString + '\n' + dataString + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()

    return output


def getPosition(sensors, datas, specifiedDatas):


    output = []

    dataRange, dataString = startStopData(datas, specifiedDatas)
    startSensors, sensorString = startSensor(sensors)

    for sensor in startSensors:
        if (sensor != -1):
            for data in dataRange:
                sensorOffset = 22*sensor+data+defaultOffset
                output.append(sensorOffset)

    history = 'getPosition: \n' + sensorString + '\n' + dataString + '\n'
    fd = open('History.txt','a')
    fd.write(history)
    fd.close()
    return output


def startSensor(sensors):
    sensorBools = [False, False, False, False, False, False, False, False, False, False, False, False, False, False]

    sensorString = 'Sensoren: '
    startSensors = np.zeros(len(sensors))


    for sensor, i in zip(sensors, range(0, len(sensors))):
        if (sensor == 'RNS' and sensorBools[0] == False):
            sensorString = sensorString + sensor + ' '
            print 'RNS'
            startSensors[i] = 6
            sensorBools[0] = True
        elif (sensor == 'RLA' and sensorBools[1] == False):
            sensorString = sensorString + sensor + ' '
            print 'RLA'
            startSensors[i] = 5
            sensorBools[1] = True
        elif (sensor == 'RUA' and sensorBools[2] == False):
            sensorString = sensorString + sensor + ' '
            print 'RUA'
            startSensors[i] = 4
            sensorBools[2] = True
        elif (sensor == 'STE' and sensorBools[3] == False):
            sensorString = sensorString + sensor + ' '
            print 'STE'
            startSensors[i] = 0
            sensorBools[3] = True
        elif (sensor == 'LUA' and sensorBools[4] == False):
            sensorString = sensorString + sensor + ' '
            print 'LUA'
            startSensors[i] = 1
            sensorBools[4] = True
        elif (sensor == 'LLA' and sensorBools[5] == False):
            sensorString = sensorString + sensor + ' '
            print 'LLA'
            startSensors[i] = 2
            sensorBools[5] = True
        elif (sensor == 'LNS' and sensorBools[6] == False):
            sensorString = sensorString + sensor + ' '
            print 'LNS'
            startSensors[i] = 3
            sensorBools[6] = True
        elif (sensor == 'RUF' and sensorBools[7] == False):
            sensorString = sensorString + sensor + ' '
            print 'RUF'
            startSensors[i] = 13
            sensorBools[7] = True
        elif (sensor == 'RLL' and sensorBools[8] == False):
            sensorString = sensorString + sensor + ' '
            print 'RLL'
            startSensors[i] = 12
            sensorBools[8] = True
        elif (sensor == 'RUL' and sensorBools[9] == False):
            sensorString = sensorString + sensor + ' '
            print 'RUL'
            startSensors[i] = 11
            sensorBools[9] = True
        elif (sensor == 'CEN' and sensorBools[10] == False):
            sensorString = sensorString + sensor + ' '
            print 'CEN'
            startSensors[i] = 7
            sensorBools[10] = True
        elif (sensor == 'LUL' and sensorBools[11] == False):
            sensorString = sensorString + sensor + ' '
            print 'LUL'
            startSensors[i] = 8
            sensorBools[11] = True
        elif (sensor == 'LLL' and sensorBools[12] == False):
            sensorString = sensorString + sensor + ' '
            print 'LLL'
            startSensors[i] = 9
            sensorBools[12] = True
        elif (sensor == 'LUF' and sensorBools[13] == False):
            sensorString = sensorString + sensor + ' '
            print 'LUF'
            startSensors[i] = 10
            sensorBools[13] = True
        else:
            startSensors[i] = -1
            print 'Keine gueltige Eingabe oder doppelte Eingabe (sensors)'
    return startSensors, sensorString

#Sucht Start und Stopp in den Daten
def startStopData(datas, specifiedDatas):
    qua = False
    acc = False
    gyr = False
    mag = False
    rE1 = False
    rE2 = False
    rE3 = False

    dataRange = []

    string = 'Daten: '
    specifiedString = 'Specified: '

    for data, i in zip(datas, range(0, len(datas))):
        if (data == 'acc' and acc == False):
            for specifiedData in specifiedDatas:
                if (specifiedData == 'x'):
                    dataRange.append(0)
                elif (specifiedData == 'y'):
                    dataRange.append(1)
                elif (specifiedData == 'z'):
                    dataRange.append(2)
            acc = True;
            string = string + data + ' '
        elif (data == 'gyr' and gyr == False):
            for specifiedData in specifiedDatas:
                if (specifiedData == 'x'):
                    dataRange.append(3)
                elif (specifiedData == 'y'):
                    dataRange.append(4)
                elif (specifiedData == 'z'):
                    dataRange.append(5)
            string = string + data + ' '
            gyr = True

        elif (data == 'mag' and mag == False):
            for specifiedData in specifiedDatas:
                if (specifiedData == 'x'):
                    dataRange.append(6)
                elif (specifiedData == 'y'):
                    dataRange.append(7)
                elif (specifiedData == 'z'):
                    dataRange.append(8)
            string = string + data + ' '
            mag = True
        elif (data == 'qua' and qua == False):
            for specifiedData in specifiedDatas:
                if(specifiedData == 'w'):
                    dataRange.append(9)
                elif (specifiedData == 'x'):
                    dataRange.append(10)
                elif (specifiedData == 'y'):
                    dataRange.append(11)
                elif (specifiedData == 'z'):
                    dataRange.append(12)
            string = string + data + ' '
            qua = True
        elif (data == 'rE1' and rE1 == False):
            for specifiedData in specifiedDatas:
                if (specifiedData == 'x'):
                    dataRange.append(13)
                elif (specifiedData == 'y'):
                    dataRange.append(14)
                elif (specifiedData == 'z'):
                    dataRange.append(15)
            string = string + data + ' '
            rE1 = True

        elif (data == 'rE2' and rE2 == False):
            for specifiedData in specifiedDatas:
                if (specifiedData == 'x'):
                    dataRange.append(16)
                elif (specifiedData == 'y'):
                    dataRange.append(17)
                elif (specifiedData == 'z'):
                    dataRange.append(18)
            string = string + data + ' '
            rE2 = True
        elif (data == 'rE3' and rE3 == False):
            for specifiedData in specifiedDatas:
                if (specifiedData == 'x'):
                    dataRange.append(19)
                elif (specifiedData == 'y'):
                    dataRange.append(20)
                elif (specifiedData == 'z'):
                    dataRange.append(21)
            string = string + data + ' '
            rE3 = True

        else:
            print 'Keine gueltige Eingabe oder doppelte Eingabe (datas)'
    return dataRange, string
