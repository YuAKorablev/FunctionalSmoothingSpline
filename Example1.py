import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FunctionalSmoothingSpline import *

if __name__ == '__main__':

    filename = "./DiscrSignals.csv"
    MyData = pd.read_csv(filename, sep=";", decimal=',')
    print(MyData)

    t_f = pd.to_datetime(MyData.t_f.dropna(), format='%d.%m.%Y')
    t_df = pd.to_datetime(MyData.t_df.dropna(), format='%d.%m.%Y')
    t_d2f = pd.to_datetime(MyData.t_d2f.dropna(), format='%d.%m.%Y')
    t_int_a = pd.to_datetime(MyData.t_int_a.dropna(), format='%d.%m.%Y')
    t_int_b = pd.to_datetime(MyData.t_int_b.dropna(), format='%d.%m.%Y')
    t_start = min(t_f[0], t_df[0], t_d2f[0], t_int_a[0])
    t_f = np.array([(x-t_start).days for x in t_f])
    t_df = np.array([(x-t_start).days for x in t_df])
    t_d2f = np.array([(x-t_start).days for x in t_d2f])
    t_int_a = np.array([(x-t_start).days for x in t_int_a])
    t_int_b = np.array([(x-t_start).days for x in t_int_b])

    y_f = MyData.y_f.dropna().to_numpy()
    y_df = MyData.y_df.dropna().to_numpy()
    y_d2f = MyData.y_d2f.dropna().to_numpy()
    y_int = MyData.y_int.dropna().to_numpy()

    m = round(3 * (len(t_f) + len(t_df) + len(t_d2f) + len(t_int_a)))

    r = FunctionalSmoothingSpline(t_f=t_f,
                                  values_f=y_f,
                                  t_df=t_df,
                                  values_df=y_df,
                                  t_d2f=t_d2f,
                                  values_d2f=y_d2f,
                                  t_int_a=t_int_a,
                                  t_int_b=t_int_b,
                                  values_int=y_int,
                                  knots_number=m,
                                  alpha=10 ** 1,
                                  All_Positive=False,
                                  info=True)

    x = r['x']
    y = r['y']
    plt.plot(x, y, color="red")
    plt.show()
    print(r)
