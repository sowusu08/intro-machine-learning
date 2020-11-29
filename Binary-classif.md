# Plot 2D-feature data to be classified
```
import matplotlib.pyplot as plt
%matplotlib inline

pos = y==1
neg = y==0

plt.plot(X[pos, 0], X[pos, 1], 'k+')
plt.plot(X[neg, 0], X[neg, 1], 'wo', mec='k')

plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(['Positive Class', 'Negative Class'])
```  
# Advanced Optimization for Logistic Reg
![](http://mathurl.com/render.cgi?g%28z%29%3D%5Cfrac%7B1%7D%7B1+e%5E%7B-z%7D%7D%5Cnocache)
