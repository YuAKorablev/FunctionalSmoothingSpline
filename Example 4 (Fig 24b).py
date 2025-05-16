import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FunctionalSmoothingSpline import *

filename = "./Sales1.csv"
MyData = pd.read_csv(filename, sep = ";", decimal=',')
print(MyData)

t = pd.to_datetime(MyData.t.dropna(), format='%d.%m.%Y')
t_start = min(t)
t = np.array([(x-t_start).days for x in t])
Y = MyData.Values.dropna().to_numpy()
n = len(t)
Y = Y[0:(n-1)]
m = round(3*n)

y = FunctionalSmoothingSpline(t_int_a = t[0:(n-1)],
                		t_int_b = t[1:n],
                		values_int = Y,
                		knots = t,
                		alpha = 10**5,
                		All_Positive = True,
                		info = False,
						add_condition_without_knots = True)

x = np.append(np.arange(t[0],t[-1],1), t[-1])

plt.plot(x, y, color = "red")
x2 = np.array([])
y2 = np.array([])
for i in range(n-1):
    x2 = np.append(x2,[ t[i], t[i+1] ])
    y2 = np.append(y2, [ Y[i]/(t[i+1]-t[i]), Y[i]/(t[i+1]-t[i]) ] )
plt.plot(x2,y2,color="grey")
plt.xticks(ticks=t, labels=[(t_start+pd.Timedelta(tx,"d")).date() for tx in t],
           rotation='vertical')
plt.show()
