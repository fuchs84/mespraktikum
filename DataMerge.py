__author__ = 'Sebastian'
from __builtin__ import int
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import pandas as pd
import math
import pylab as plb
import pandas as pd
import computeAngles as ca

#daten laden die benoetigt werden
x1bus1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\ProbandenWalk\ID004\20151126\x1\xbus.csv', sep = "\t")
x1bus2 = pd.read_csv(r'C:\Users\Sebastian\Desktop\ProbandenWalk\ID004\201511262\x1\xbus.csv', sep = "\t")

x2bus1 = pd.read_csv(r'C:\Users\Sebastian\Desktop\ProbandenWalk\ID004\20151126\x2\xbus.csv', sep = "\t")
x2bus2 = pd.read_csv(r'C:\Users\Sebastian\Desktop\ProbandenWalk\ID004\20151126\x2\xbus.csv', sep = "\t")




#speichern der daten als dataframes
datax11 = pd.DataFrame(x1bus1.as_matrix())
datax12 = pd.DataFrame(x1bus2.as_matrix())
datax21 = pd.DataFrame(x2bus1.as_matrix())
datax22 = pd.DataFrame(x2bus2.as_matrix())
#datax23 = pd.DataFrame(x2bus3.as_matrix())


#aneinanderfuegen von daten des gleichen sensorsets
x1bus = pd.concat([datax11,datax12])
x2bus = pd.concat([datax21,datax22])

#wieder als matrix speichern //nicht sicher ob das dauernde formataendern wirklich noetig ist
datax1 = x1bus.as_matrix()
datax2 = x2bus.as_matrix()

#Die Zeit und Duration merger Methode von Matthias
def merger(dataSet):
    time = dataSet[:, 0]
    duration = dataSet[:, 1]
    for a in range(0, len(time)):
        tempDuration = str(int(duration[a]))
        tempTime = str(int(time[a]))
        if (len(tempDuration) < 6):
            for b in range(len(tempDuration), 6):
                tempDuration = '0' + tempDuration
        tempDuration = tempDuration[0:2]
        dataSet[a,0] = float(tempTime + tempDuration)
    return dataSet

#Daten in das richtige Format bringen
#erstellen von einzigartigen Zeitstempel als Index um aneinanderfuegen zu ermoeglichen
#nicht moeglich mit Time und duration gemerged, da hier im kleinen Bereich unterschiede auftreten, z.B eine Serie ...01,...03,...05 und die andere...02,...04,...06 etc
#indmerge = merger(datax1)
#inmerge2 = merger(datax2)
newDatax1 = (datax1)
index1 = []
a=0
for i in range(0,len(newDatax1[:,0])-1):
    if  newDatax1[i,0] == newDatax1[i+1,0]:
        a = a+1
        add = newDatax1[i,0]+(a*0.00001)
        index1.append(add)
    else:
        a=a+1
        add = newDatax1[i,0]+a*0.00001
        index1.append(add)
        a=1
a=100
index1.append(newDatax1[len(newDatax1)-1,0]+(a*0.00001))
a=0

#erstellen von einzigartigen Zeitsempel
newDatax2 = (datax2)
index2 = []
for i in range(0,len(newDatax2[:,0])-1):
    if newDatax2[i,0] == newDatax2[i+1,0]:
        a = a+1
        add = newDatax2[i,0]+(a*0.00001)
        index2.append(add)

    else:
        a=a+1
        add = newDatax2[i,0]+(a*0.00001)
        index2.append(add)
        a=1
a=100
index2.append(newDatax2[len(newDatax2)-1,0]+(a*0.00001))



#verwenden der generierten einzigartigen indices mit datensets
dfx1 = pd.DataFrame(newDatax1, index=index1)
dfx2 = pd.DataFrame(newDatax2[:,2:], index=index2)
datas = [dfx1, dfx2]

#vereinigen entlang der x-Achse ohne Zeilen mit leeren Werten:
#WICHTIG: join = 'inner'  anfuegen um keine Zeilen mit null also nur zeiten in denen beide werte haben zu erkennen
mergedtime = pd.concat(datas, axis=1)
mergedtimefiller = pd.concat([dfx1, pd.DataFrame(newDatax2, index=index2)], axis=1)
print "merged"
#nan durch 0 ersetzen
mergedtime = mergedtime.fillna(0)
mergedtimefiller = pd.DataFrame(mergedtimefiller.iloc[:,93:95])
for i in range(0,len(mergedtime)):
    if mergedtime.iloc[i,0] == 0:
        mergedtime.iloc[i,0] = mergedtimefiller.iloc[i,0]
        mergedtime.iloc[i,1] = mergedtimefiller.iloc[i,1]
print "null filled"

tempmergedtime = mergedtime.reset_index(drop= True)
#fuellen der leeren zeiten mit vortlaufenden Zeitstempeln und erstellter Duration
counter =0
emptypos = []

#ermitteln der Postition der fuellenden Luecken
for i in range(1,len(tempmergedtime)):
    distance = tempmergedtime.iloc[i,0]-tempmergedtime.iloc[i-1,0]
    start = tempmergedtime.iloc[i-1,0]
    if (distance) > 1:
        counter+=1
        emptypos.append(i-1)
emptypos.append(len(tempmergedtime)-1)
emptylist = []
#generieren der Lueckenfueller
for i in range(1,len(tempmergedtime)):
        distance = tempmergedtime.iloc[i,0]-tempmergedtime.iloc[i-1,0]
        start = tempmergedtime.iloc[i-1,0]
        if (distance) > 1:
            nframe = np.zeros(shape=(((distance-1)*50),len(tempmergedtime.iloc[0,:])))
            empty = pd.DataFrame(nframe)
            for j in range(0,len(empty),50):
                start +=1
                duration = 12213
                for k in range(0,50):
                    empty.iloc[k+j,0]= start
                    empty.iloc[k+j,1]= duration
                    duration += 20000

            emptylist.append(empty)
print "mergeok"
mergerino = pd.DataFrame(tempmergedtime.iloc[0:emptypos[0],:])
print mergerino.iloc[:,93:95]
for i in range(0,counter):
    addfiller = pd.DataFrame(emptylist[i])
    for k in range(0,len(addfiller)):
        mergerino.loc[len(mergerino)+k,:] = addfiller.iloc[k,:]
    test = tempmergedtime.iloc[emptypos[i]+1:emptypos[i+1],:]
    mergerino = mergerino.append(test,ignore_index =True)

print "keine zeit-Nullen"
#in echter zeit, nicht sicher ob benoetigt, auskommentieren falls dies der fall ist
# time = pd.DataFrame(mergedtime)
# getMillis = pd.DataFrame(merger(time.as_matrix()))
# for i in range(1, len(mergedtime)):
#     mergedtime.iloc[i-1,0]=   datetime.datetime.fromtimestamp(
#                                 int(mergedtime.iloc[i-1,0])
#                             ).strftime('%Y-%m-%d %H:%M:%S')
#     mergedtime.iloc[i-1,0]+=(":")
#     millis = (str(getMillis.iloc[i-1,0])[10:12])
#     print millis
#     mergedtime.iloc[i-1,0]+=(millis)

#abspeichern als csv
#mergedtime.to_csv('mergeddata.csv',sep='\t', index= False, header= False)
print "exited"
savearray = np.array(mergerino)

rotatedVectors = ca.computeRotation(savearray)

output = savearray[:, :2]
numberofsensors = 14
for i in range(0, numberofsensors):
    startData = 2 + i * 13
    stopData = 2 + i * 13 + 13
    startVectors = i * 3
    stopVectors = i * 3 + 3
    print ('startData: %d stopData: %d' %(startData, stopData))
    print ('startVectors: %d stopVectors: %d' %(startVectors, stopVectors))

    output = np.c_[output, savearray[:, startData:stopData], rotatedVectors[:, startVectors:stopVectors, 0],
                   rotatedVectors[:, startVectors:stopVectors, 1],rotatedVectors[:, startVectors:stopVectors, 2]]

    print rotatedVectors.shape
    print ('output.shape')
    print output.shape
    print output[1,:].shape

np.savetxt('mergedID001.csv', fmt=['%i','%i','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f','%f',
                              '%f','%f','%f','%f'] ,X= output, delimiter='\t')
print "saved"


#plots zeigen zum vergleich, keine eigentliche relevanz
plt.subplot(4, 1, 1)
plt.plot(mergerino.iloc[:,2:93])

plt.subplot(4, 1, 2)
plt.plot(dfx1.iloc[:,9:])


plt.subplot(4, 1, 3)
plt.plot(dfx2.iloc[:,3:])

plt.subplot(4, 1, 4)
plt.plot(mergerino.iloc[500:,93:])

plt.show()