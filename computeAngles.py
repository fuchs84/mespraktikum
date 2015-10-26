__author__ = 'Matthias'

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd



def computeRotation(dataSet):
    quaternions = searchQuaternionInData(dataSet)
    rotatedVectors = rotationQuaternion(quaternions)
    np.savetxt('rotatedVectorsE1.csv', rotatedVectors[:, :, 0], delemiter='\t')
    np.savetxt('rotatedVectorsE2.csv', rotatedVectors[:, :, 1], delemiter='\t')
    np.savetxt('rotatedVectorsE3.csv', rotatedVectors[:, :, 2], delemiter='\t')



# Sucht die Quaternions aus einem Datenset heraus.
def searchQuaternionInData(dataSet):
    numberOfSensors = (len(dataSet[0, :])-2)/13
    quaternions = np.zeros(shape=(len(dataSet[:,0]), 0))
    for i in range(0,numberOfSensors):
        startQuaternion = i*13+11
        stopQuarternion = i*13+15
        quaternions = np.c_[quaternions, dataSet[:, startQuaternion:stopQuarternion]]
    return quaternions

# Berechnet die Rotationsmatrix eines Quaternions.
def rotationMatrixQuaternion(quaternion):
    a, b, c ,d = quaternion

    r11 = (a**2+b**2-c**2-d**2)
    r12 = 2.0*(b*c-a*d)
    r13 = 2.0*(b*d+a*c)

    r21 = 2.0*(b*c+a*d)
    r22 = a**2-b**2+c**2-d**2
    r23 = 2.0*(c*d-a*b)

    r31 = 2.0*(b*d-a*c)
    r32 = 2.0*(c*d+a*b)
    r33 = a**2-b**2-c**2+d**2

    rotationMatrix = np.matrix([[r11, r12, r13],[r21, r22, r23],[r31, r32, r33]])
    return rotationMatrix

# Rotiert die drei Einheitsvektoren (e1, e2, e3) mit den Quaternions
def rotationQuaternion(quaternions):
    e1 = np.matrix([[1.0], [0.0], [0.0]])
    e2 = np.matrix([[0.0], [1.0], [0.0]])
    e3 = np.matrix([[0.0], [0.0], [1.0]])

    numberOfSensors = len(quaternions[0, :])/4
    rotatedVectors = np.zeros(shape=(len(quaternions[:,0]), numberOfSensors*3, 3))
    for i in range(1, len(quaternions[:, 0])):
        for j in range(0, numberOfSensors):
            start = j*4
            stop = j*4+4
            rotationMatrix = rotationMatrixQuaternion(quaternions[i, start:stop])
            rotateE1 = rotationMatrix*e1
            rotateE2 = rotationMatrix*e2
            rotateE3 = rotationMatrix*e3

            start = j*3
            stop = j*3+3
            rotatedVectors[i,start:stop, 0] = np.transpose(rotateE1)
            rotatedVectors[i,start:stop, 1] = np.transpose(rotateE2)
            rotatedVectors[i,start:stop, 2] = np.transpose(rotateE3)

            #rotatedVectors[i, 0:3, j] = np.transpose(rotateE1)
            #rotatedVectors[i, 3:6, j] = np.transpose(rotateE2)
            #rotatedVectors[i, 6:9, j] = np.transpose(rotateE3)
    return rotatedVectors


def computeAngles(s1, s2, rotatedVectors):
    angle = np.zeros(shape=(len(rotatedVectors[:, 0, 0]), 3))
    for i in range (0, len(rotatedVectors[:, 0, 0])):
        for j in range(0, 3):
            x = j*3
            y = j*3 + 1
            z = j*3 + 2
            vectorS1Norm = math.sqrt(rotatedVectors[i, x, s1]**2 + rotatedVectors[i, y, s1]**2 + rotatedVectors[i, z, s1]**2)
            vectorS2Norm = math.sqrt(rotatedVectors[i, x, s2]**2 + rotatedVectors[i, y, s2]**2 + rotatedVectors[i, z, s2]**2)


            scalarProduct = rotatedVectors[i, x, s1]*rotatedVectors[i, x, s2] +rotatedVectors[i, y, s1]*rotatedVectors[i, y, s2] + rotatedVectors[i, z, s1]*rotatedVectors[i, z, s2]
            singleAngle = math.degrees(math.acos(scalarProduct/(vectorS1Norm*vectorS2Norm)))
            angle[i, j] = singleAngle
    return angle

#Vertauscht die Vektoren von zwei Sensoren
def swapRotatedVectors(s1, s2, rotatedVectors):
    temp = rotatedVectors[:, : ,s1]
    rotatedVectors[:, :, s1] = rotatedVectors[:, :, s2]
    rotatedVectors[:, :, s2] = temp
    return rotatedVectors

def computeRotation(dataSet):
    quaternions = searchQuaternionInData(dataSet)
    rotatedVectors = rotationQuaternion(quaternions)
    np.savetxt('rotatedVectorsE1.csv', rotatedVectors[:, :, 0], delimiter='\t')
    np.savetxt('rotatedVectorsE2.csv', rotatedVectors[:, :, 1], delimiter='\t')
    np.savetxt('rotatedVectorsE3.csv', rotatedVectors[:, :, 2], delimiter='\t')
