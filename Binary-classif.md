## Plot 2D-feature data to be classified
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
<img src="https://embed.deepnote.com/95db573e-f8e2-4b3b-9132-23fcef5280a2/72126a9f-6918-4819-9b01-209c2880ff83/00000-b0747369-c721-49ea-bdb6-e0a66c9ad546?height=288" height="288" width="500"/>
