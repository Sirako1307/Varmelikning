import numpy as np
import matplotlib.pyplot as plt

#constants
c=1
h=0.1
k=0.1

lengde=10
tid=10

#More consts

x=np.arange(0,10,h)
t=np.arange(0,10,k)

#Init cond
#Must do

#waves
u=np.empty((lengde,tid))

#Euler explixit

for j in range(10/0,1):
    for i in range(10/0,1):
        u[i,j+1]=(c*(h**2/k**2))(u[i+1,j]-2*u[i,j]+u[i-1,j]) +2*u[i,j] -u[i,j-1]
