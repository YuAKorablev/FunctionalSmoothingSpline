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
x = np.arange(t[0],t[-1],step = 0.1)
r = FunctionalSmoothingSpline(t_int_a = t[0:(n-1)],
							   t_int_b = t[1:n],
							   values_int = Y,
							   knots_number = m,
							   alpha = 10**5,
							   x = x,
							   All_Positive = True,
							   method = "Lemke",
							   output = "integral",
							   info = True,
							   add_knots=True,
							   new_knots_tol=10 ** (-4),
							   max_added_knots=50,
							   add_condition_without_knots=True
								)
x = r['x']
y = r['integral']
# x = np.append(np.arange(t[0],t[-1],1), t[-1])

plt.plot(x, y, color = "red")
x2 = np.array([])
y2 = np.array([])
for i in range(n-1):
    x2 = np.append(x2,[ t[i], t[i+1] ])
    y2 = np.append(y2, [ Y[i]/(t[i+1]-t[i]), Y[i]/(t[i+1]-t[i]) ] )
plt.plot(x2,y2,color="grey")
plt.xticks(ticks=t, labels=[(t_start+pd.Timedelta(tx,"d")).date() for tx in t],
           rotation='vertical')

old_knots = r['knots']
new_knots = r['new_knots']
old_knots = np.setdiff1d(old_knots, new_knots)
print("new_knots = ", new_knots)
print("len(new_knots) = ", len(new_knots))
plt.scatter(old_knots, np.zeros(len(old_knots)), marker="+", c="red")
plt.scatter(new_knots, np.zeros(len(new_knots)), marker="+", c="blue")
# plt.scatter(179.97007425886108, 0, marker="o", c="magenta")

plt.show()
