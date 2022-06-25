import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


fA=open("C:\\Users\\ilhanp\\Desktop\\a.txt" , "r",encoding="utf-8")
fB=open("C:\\Users\\ilhanp\\Desktop\\b.txt" , 'r',encoding="utf-8")
 
A=fA.readlines()[4:]
B=fB.readlines()[4:]

datalistA=[]
datalistB=[]

for line in A:
    a=line.split()
    datalistA.append([float(i) for i in a])
for line in B:
    a=line.split()
    datalistB.append([float(i) for i in a]) 

dataA=np.asarray(datalistA)
dataB=np.asarray(datalistB)
hullA=ConvexHull(dataA)
hullB=ConvexHull(dataB) 
plt.scatter(dataA[:,0],dataA[:,1],c='r')
plt.scatter(dataB[:,0],dataA[:,1],c='b')
plt.xlim([-3,5])
plt.ylim([-3,5])
plt.legend(['Class1','Class2'])
plt.plot(dataA[hullA.vertices,0],dataA[hullA.vertices,1],'r--',lw=2)
plt.plot(dataB[hullB.vertices,0],dataB[hullB.vertices,1],'b--',lw=2)
plt.show()
#Initializing the weights with normal distribution

#import matplotlib.lines as lines
W=np.random.normal(0,0.1,3)
learningrate=0.6
# Training Session
for iteration in range(51):
    foo=[]

    for i in range(100):

        if np.dot(W,(-1)*np.append(dataA[i],1)) >= 0 :
            data=np.append(dataA[i],1) 
            foo.append((-1)*data)
        if np.dot(W,np.append(dataB[i],1)) >= 0 :
            data=np.append(dataB[i],1) 
            foo.append(data)
    foo=np.asarray(foo)
    loss=np.sum(foo,axis=0)
    update=learningrate*loss 
    learningrate=learningrate-0.01 
    W=W-update

    print ("iteration number:", iteration)
    print ("number of wrong predictions:",foo.shape[0]) 
    print ("weights ore:",W)


r=np.array(range(-4,10)) 
y=(-1)*(W[0]/W[1])*r-W[2]/W[1]

plt.scatter(dataA[:,0],dataA[:,1],c='r') 
plt.scatter(dataB[:,0],dataB[:,1],c='b') 
plt.xlim([-3,5])
plt.ylim([-3,5])
plt.legend(['Class1','Class2'])
plt.plot(r,y)
plt.show()
