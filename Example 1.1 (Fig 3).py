import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FunctionalSmoothingSpline import *

if __name__ == '__main__':

    filename = "./Data1.csv"
    MyData = pd.read_csv(filename, sep=";", decimal=',')
    print(MyData)

    t_f = pd.to_datetime(MyData.t_f.dropna(), format='%d.%m.%Y')
    # t_df = pd.to_datetime(MyData.t_df.dropna(), format='%d.%m.%Y')
    # t_d2f = pd.to_datetime(MyData.t_d2f.dropna(), format='%d.%m.%Y')
    #t_int_a = pd.to_datetime(MyData.t_int_a.dropna(), format='%d.%m.%Y')
    #t_int_b = pd.to_datetime(MyData.t_int_b.dropna(), format='%d.%m.%Y')
    #t_start = min(min(t_f), min(t_df), min(t_d2f) )#, t_int_a)
    t_start = min(t_f)
    t_f = np.array([(x-t_start).days for x in t_f])
    # t_df = np.array([(x-t_start).days for x in t_df])
    # t_d2f = np.array([(x-t_start).days for x in t_d2f])
    #t_int_a = np.array([(x-t_start).days for x in t_int_a])
    #t_int_b = np.array([(x-t_start).days for x in t_int_b])

    y_f = MyData.y_f.dropna().to_numpy()
    # y_df = MyData.y_df.dropna().to_numpy()
    # y_d2f = MyData.y_d2f.dropna().to_numpy()
    #y_int = MyData.y_int.dropna().to_numpy()

    #m = round(3 * (len(t_f) + len(t_df) + len(t_d2f) ))#+ len(t_int_a)))
    m = round(3 * len(t_f))
    # x = np.arange(38,46,step = 0.0001)
    r = FunctionalSmoothingSpline(t_f=t_f,
                                  values_f = np.log(y_f),
                                  #t_df=t_df,
                                  #values_df=y_df,
                                  #t_d2f=t_d2f,
                                  #values_d2f=y_d2f,
                                  #t_int_a=t_int_a,
                                  #t_int_b=t_int_b,
                                  #values_int=y_int,
                                  knots_number=m,
                                  alpha=10 ** 2 / 2.3,
                                  # x = x,
                                  All_Positive=False,
                                  # output = ["y","dy","d2y","integral"],
                                  info=True,
                                   )

    x = r['x']
    y = r['y']
    plt.plot(x, np.exp(y), color="red")
    # plt.plot(x,  r['dy'], color="orange")
    # plt.plot(x, r['d2y'], color="magenta")

    #t = np.concatenate ((t_f, t_df ))
    t = t_f
    plt.xticks(ticks=t, labels=[(t_start + pd.Timedelta(tx, "d")).date() for tx in t],
               rotation='vertical')
    plt.scatter(t_f,y_f)
    # plt.show()

    # plt.plot(x, r['integral'], color="red")

    # old_knots = r['knots']
    # new_knots = r['new_knots']
    # old_knots = np.setdiff1d(old_knots,new_knots)
    # print("new_knots = ", new_knots)
    # print("len(new_knots) = ", len(new_knots))
    # plt.scatter(old_knots, np.zeros(len(old_knots)), marker="+", c = "red")
    # plt.scatter(new_knots,np.zeros(len(new_knots)), marker = "+", c ="blue")
    #plt.scatter(t_df, y_df)
    #plt.scatter(t_d2f, y_d2f)
    plt.show()
    #print(r)





