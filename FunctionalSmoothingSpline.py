#                      Functional Smoothing cubic spline
#   Restoration of unknown function by values, first derivatives, second derivatives and integrals with cubic smoothing spline in Python.
#   The method minimizes 4 sums of squares of the difference between:
# 1) observed values and values of the restored function
# 2) observed first derivatives and first derivatives of the restored function
# 3) observed second derivatives and second derivatives of the restored function
# 4) observed integrals and integrals of the restored function
# plus nonlinearity (roughness) penalty of the spline.
#
#    Autor - Korablev Yuriy Aleksandrovich
# PhD in Economics, assistant professor, department "System Analysis in Economics"
# Financial University under the Government of the Russian Federation
# email: yura-korablyov@yandex.ru, YUAKorablev@fa.ru

# Copyright © 2024  Korablev Yuriy Aleksandrovich
#   
#   This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>


import numpy as np
from numba import njit,types
from MyLemke import *
import cvxopt
import scipy.optimize as optimize

def FunctionalSmoothingSpline(
        t_f=None,  # array of observation moments
        values_f=None,  # array of observation values
        weights_f=None,  # array of  weights
        t_df=None,  # array of first derivative moments
        values_df=None,  # array of first derivative values
        weights_df=None,  # array of first derivative weights
        coef_df=1,  # coefficient of first derivative sum of squares
        t_d2f=None,  # array of second derivative moments
        values_d2f=None,  # array of second derivative values
        weights_d2f=None,  # array of second derivative weights
        coef_d2f=1,  # coefficient of second derivative sum of squares
        t_int_a=None,  # array of integrals start moments
        t_int_b=None,  # array of integrals end moments
        values_int=None,  # array of integral values
        weights_int=None,  # array of integral weights
        coef_int=1,  # coefficient of integral sum of squares
        knots=None,  # knots
        knots_number=None,  # number of knots
        alpha=1,  # smoothing parameter
        x=None,  # output moments
        All_Positive=False,  # solve as monotone spline by Lemke algorithm
        method="Lemke",    # method for solving a quadratic programming problem
        output = ["y"],    # values to output, list ["y", "dy", "d2y", "integral"]
        # integral=False,  # return integral function F(t)
        info=False,     # need info?
        add_knots = False, # add new knots when function below zero
        add_condition_without_knots = False,
        new_knots_tol = 0.0001, # tolerans to add new knots
        max_added_knots = 10
):

    if knots_number is None and knots is None:
        knots_number = 0
        if t_f is not None and len(t_f) > 0:
            knots_number = len(t_f)
        if t_df is not None and len(t_df) > 0 and knots_number < len(t_df):
            knots_number = len(t_df)
        if t_d2f is not None and len(t_d2f) > 0 and knots_number < len(t_d2f):
            knots_number = len(t_d2f)
        if t_int_a is not None and len(t_int_a) > 0 and knots_number < len(t_int_a):
            knots_number = len(t_int_a)

    if knots is not None:
        knots_number = len(knots)

    assert knots_number >= 2, 'knots_number or observations should not be less than 2'

    # in case knots is not defined
    if knots is None or knots_number != len(knots):  # when knots_number defined, but knots not defined
        start_knot = + np.inf
        end_knot = - np.inf
        if t_f is not None and len(t_f) > 0:
            if start_knot > np.min(t_f):
                start_knot = np.min(t_f)
            if end_knot < np.max(t_f):
                end_knot = np.max(t_f)
        if t_df is not None and len(t_df) > 0:
            if start_knot > np.min(t_df):
                start_knot = np.min(t_df)
            if end_knot < np.max(t_df):
                end_knot = np.max(t_df)
        if t_d2f is not None and len(t_d2f) > 0:
            if start_knot > np.min(t_d2f):
                start_knot = np.min(t_d2f)
            if end_knot < np.max(t_d2f):
                end_knot = max(t_d2f)
        if t_int_a is not None and len(t_int_a) > 0 and len(t_int_b) > 0:
            if start_knot > np.min(t_int_a):
                start_knot = np.min(t_int_a)
            if end_knot < np.max(t_int_b):
                end_knot = np.max(t_int_b)
        knots = np.linspace(start_knot, end_knot, knots_number)

    need_repeat = True;  # when we added new knots we neeed return and recalculate everything
    new_knots_global = []
    Z = []               # extra conditions will be stored here

    while need_repeat:
        m = knots_number  # for short

        h = np.zeros(m - 1)  # array of distance between knots
        h[0:(m - 1)] = knots[1:m] - knots[0:(m - 1)]

        # Matrix Q
        Q = np.zeros((m, m - 2))
        for i in range(m - 2):
            Q[i, i] = 1 / h[i];
            Q[i + 1, i] = - 1 / h[i] - 1 / h[i + 1];
            Q[i + 2, i] = 1 / h[i + 1]

        # Matrix R
        R = np.zeros((m - 2, m - 2))
        for i in range(m - 2):
            R[i, i] = (h[i] + h[i + 1]) / 3;
            if (i < m - 2 - 1):
                R[i + 1, i] = h[i + 1] / 6;
                R[i, i + 1] = h[i + 1] / 6;

        # Matrix K calculation
        inv_R = np.linalg.inv(R)
        t_Q = np.transpose(Q)
        Rm1QT = inv_R @ t_Q
        K = Q @ Rm1QT
        # K = Q @ inv_R @ t_Q

        # ========= 1. Observation (t_f, values_f)  ===========

        if t_f is not None and len(t_f) > 0:
            nf = len(t_f)  # number of observation coordinates
            assert len(values_f) == nf, 'length of values_f and t_f must be same'

            if weights_f is None:
                weights_f = np.ones(nf)
            assert len(weights_f) == nf, 'length of weights_f and t_f must be same'
            Wf = np.diag(weights_f)

            # reorder observations (t_f, values_f) by appear time t_f
            ord = np.argsort(t_f)
            t_f = t_f[ord]
            values_f = values_f[ord]
            values_f = values_f.reshape((nf, 1))

            # Filling in Vf and Pf matrices
            Vf = np.zeros((nf, m))
            Pf = np.zeros((nf, m))
            k = 0  # start knot
            for i in range(nf):
                while knots[k] <= t_f[i] and knots[k + 1] < t_f[
                    i] and k < knots_number:  # find first k, that knots[k+1]>t_f[i]
                    k += 1
                hk_m = t_f[i] - knots[k]
                hk_p = knots[k + 1] - t_f[i]
                Vf[i, k] = hk_p / h[k]
                Vf[i, k + 1] = hk_m / h[k]
                Pf[i, k] = hk_m * hk_p * (h[k] + hk_p) / (6 * h[k])
                Pf[i, k + 1] = hk_m * hk_p * (h[k] + hk_m) / (6 * h[k])
            Pf = Pf[:, 1:(m - 1)]  # don't need first and last column

            # Matrix Cf calculation
            Cf = Vf - Pf @ inv_R @ t_Q
            t_Cf = np.transpose(Cf)

        # ========= 2. Observation (t_df, values_df)  ===========

        if t_df is not None and len(t_df) > 0:
            ndf = len(t_df)  # number of observation
            assert len(values_df) == ndf, 'length of values_df and t_df must be same'

            if weights_df is None:
                weights_df = np.ones(ndf)
            assert len(weights_df) == ndf, 'length of weights_df and t_df must be same'
            Wdf = np.diag(weights_df)

            ord = np.argsort(t_df)  # reorder observations (t_df, values_df)  by appear time t_df
            t_df = t_df[ord]
            values_df = values_df[ord]
            values_df = values_df.reshape((ndf, 1))

            # Filling in Vdf and Pdf matrices
            Vdf = np.zeros((ndf, m))
            Pdf = np.zeros((ndf, m))
            k = 0  # start knot
            for i in range(ndf):
                while knots[k] <= t_df[i] and knots[k + 1] < t_df[i] and k < m:  # find first k, that knots[k + 1]>t_df[i]
                    k = k + 1
                hk_m = t_df[i] - knots[k]
                hk_p = knots[k + 1] - t_df[i]

                Vdf[i, k] = - 1 / h[k]
                Vdf[i, k + 1] = 1 / h[k]
                Pdf[i, k] = - h[k] / 6 + (hk_p) ** 2 / (2 * h[k])
                Pdf[i, k + 1] = h[k] / 6 - (hk_m) ** 2 / (2 * h[k])
            Pdf = Pdf[:, 1:(m - 1)]  # don't need first and last column

            # Matrix Cdf calculation
            Cdf = Vdf - Pdf @ inv_R @ t_Q
            t_Cdf = np.transpose(Cdf)

        # ========= 3. Observation (t_d2f, values_d2f)  ===========

        if t_d2f is not None and len(t_d2f) > 0:
            nd2f = len(t_d2f)  # number of observation
            assert len(values_d2f) == nd2f, 'length of values_d2f and t_d2f must be same'

            if weights_d2f is None:
                weights_d2f = np.ones(nd2f)
            assert len(weights_d2f) == nd2f, 'length of weights_d2f and t_d2f must be same'
            Wd2f = np.diag(weights_d2f)
            # reorder observations (t_d2f, values_d2f)  by appear time t_d2f
            ord = np.argsort(t_d2f)
            t_d2f = t_d2f[ord]
            values_d2f = values_d2f[ord]
            values_d2f = values_d2f.reshape((nd2f, 1))

            # Filling in Pd2f matrices
            Pd2f = np.zeros((nd2f, m))
            k = 0  # start knot
            for i in range(nd2f):
                while knots[k] <= t_d2f[i] and knots[k + 1] < t_d2f[i] and k < m:  # find first k, that knots[k+1]>t_d2f[i]
                    k = k + 1
                hk_m = t_d2f[i] - knots[k]
                hk_p = knots[k + 1] - t_d2f[i]

                Pd2f[i, k] = - hk_p / h[k]
                Pd2f[i, k + 1] = - hk_m / h[k]
            Pd2f = Pd2f[:, 1:(m - 1)]  # don't need first and last column

            # Matrix Cd2f calculation
            Cd2f = - Pd2f @ inv_R @ t_Q
            t_Cd2f = np.transpose(Cd2f)

        # ========= 4. Observation (t_int_a, t_int_b, values_int)  ===========

        if t_int_a is not None and len(t_int_a) > 0:
            nint = len(t_int_a)  # number of observation
            assert len(t_int_b) == nint, 'length of t_int_b and t_int_a must be same'
            assert len(values_int) == nint, 'length of values_int and t_int_a must be same'
            if weights_int is None:
                weights_int = np.ones(nint)
            assert len(weights_int) == nint, 'length of weights_int and t_int_a must be same'
            Wint = np.diag(weights_int)

            # reorder observations (t_int_a, t_int_b, values_int) by appear time t_int_a
            ord = np.argsort(t_int_a)  # order(t_int_a, t_int_b,values_int)
            t_int_a = t_int_a[ord]
            t_int_b = t_int_b[ord]
            values_int = values_int[ord]
            values_int = values_int.reshape((nint, 1))

            # Filling in Vint and Pint matrices
            Vint = np.zeros((nint, m))
            Pint = np.zeros((nint, m))
            k = 0  # start knot
            for i in range(nint):
                while knots[k] <= t_int_a[i] and knots[k + 1] < t_int_a[
                    i] and k < m:  # find first k, that knots[k + 1]>t_int_a[i]
                    k = k + 1

                    # finding L, it can be 0
                L = 0
                while t_int_b[i] > knots[k + L + 1] and k + L + 1 < m - 1:
                    L += 1

                hk_m = t_int_a[i] - knots[k]
                hk_p = knots[k + 1] - t_int_a[i]
                hkL_m = t_int_b[i] - knots[k + L]
                hkL_p = knots[k + L + 1] - t_int_b[i]

                Vint[i, k] = (hk_p) ** 2 / h[k] / 2
                Pint[i, k] = h[k] ** 3 / 24 - (hk_m) ** 2 * (hk_p + h[k]) ** 2 / h[k] / 24
                l = 1
                while l <= L:
                    Vint[i, k + l] = (h[k + l - 1] + h[k + l]) / 2
                    Pint[i, k + l] = (h[k + l - 1] ** 3 + h[k + l] ** 3) / 24
                    l += 1
                Vint[i, k + 1] = Vint[i, k + 1] - (hk_m) ** 2 / h[k] / 2
                Pint[i, k + 1] = Pint[i, k + 1] + (hk_m) ** 2 * ((hk_m) ** 2 - 2 * h[k] ** 2) / h[k] / 24
                Vint[i, k + L] = Vint[i, k + L] - (hkL_p) ** 2 / h[k + L] / 2
                Pint[i, k + L] = Pint[i, k + L] + (hkL_p) ** 2 * ((hkL_p) ** 2 - 2 * h[k + L] ** 2) / h[k + L] / 24
                Vint[i, k + L + 1] = (hkL_m) ** 2 / h[k + L] / 2
                Pint[i, k + L + 1] = h[k + L] ** 3 / 24 - (hkL_p) ** 2 * (hkL_m + h[k + L]) ** 2 / h[k + L] / 24

            Pint = Pint[:, 1:(m - 1)]  # don't need first and last column

            # Matrix Cint calculation
            Cint = Vint - Pint @ inv_R @ t_Q
            t_Cint = np.transpose(Cint)

        # ============ Calculation =============

        # matrix D
        D = alpha * K
        if t_f is not None and len(t_f) > 0:
            D = D + t_Cf @ Wf @ Cf
        if t_df is not None and len(t_df) > 0:
            D = D + coef_df * (t_Cdf @ Wdf @ Cdf)
        if t_d2f is not None and len(t_d2f) > 0:
            D = D + coef_d2f * t_Cd2f @ Wd2f @ Cd2f
        if t_int_a is not None and len(t_int_a) > 0:
            D = D + coef_int * t_Cint @ Wint @ Cint

        # matrix c
        c = np.zeros((m, 1))
        if t_f is not None and len(t_f) > 0:
            c = c + t_Cf @ Wf @ values_f
        if t_df is not None and len(t_df) > 0:
            c = c + coef_df * t_Cdf @ Wdf @ values_df
        if t_d2f is not None and len(t_d2f) > 0:
            c = c + coef_d2f * t_Cd2f @ Wd2f @ values_d2f
        if t_int_a is not None and len(t_int_a) > 0:
            c = c + coef_int * t_Cint @ Wint @ values_int

        # Calculation of g and gamma
        if All_Positive or method in ["exp"]:
            match method:
                case "Lemke":
                    if add_condition_without_knots and len(Z)>0:
                        # print("Lemke len(Z) = ", len(Z))
                        M = np.vstack((  np.hstack( (D,            -np.array(Z).transpose() ) ) ,
                                         np.hstack( ( np.array(Z), np.zeros( (len(Z),len(Z)))   ))
                                        ) )
                        q = np.vstack(( -c,
                                        np.zeros((len(Z),1))
                                       ) )
                        g, exit_code, exit_string = Lemke(M, q, maxIter=10000)
                    else:
                        g, exit_code, exit_string = Lemke(D, -c, maxIter=10000)

                    if exit_code>0:
                        print("exit_code = ", exit_code)
                        print("exit_string = ", exit_string)
                        return [np.nan]
                    g = g[0:m]
                case "Lemke_njit":
                    g, exit_code, exit_string = Lemke_njit(D, -c, maxIter=10000)
                case "cvxopt":
                    args = [cvxopt.matrix(D), cvxopt.matrix(-c)] # (D+D.transpose())/2
                    if add_condition_without_knots and len(Z)>0:
                        M = -np.eye(m)
                        # print("M.shape = ", M.shape)
                        for zi in Z:
                            M = np.vstack((M, -zi))
                        args.extend([cvxopt.matrix(M), cvxopt.matrix(np.zeros(M.shape[0]))])
                        # print("Solving with cvxopt, M.shape = ", M.shape)
                    else:
                        args.extend([cvxopt.matrix(-np.eye(m)), cvxopt.matrix(np.zeros(m))])
                        # print("Solving with cvxopt, np.eye(m).shape = ", np.eye(m).shape)
                    # print(args)
                    # options ={'abstol': 1e-12, 'reltol': 1e-12, 'feastol' : 1e-12}
                    sol = cvxopt.solvers.qp(*args) #, options = options

                    if 'optimal' not in sol['status']:
                        print(sol['status'])
                        return [np.nan]
                    else:
                        # print("cvxopt")
                        g = np.array(sol['x']).reshape((m,))
                        # if "M" in locals():
                        #     print("M @ g = ", M @ g)

                case "exp":

                    def loss(g):
                        # gamma = inv_R @ t_Q @ g
                        gamma = Rm1QT @ g
                        g2 = np.append([0], gamma)
                        g2 = np.append(g2, 0)
                        S = alpha * g.T @ K @ g
                        if t_f is not None and len(t_f) > 0:
                            k = 0
                            for i in range(nf):
                                while knots[k] <= t_f[i] and knots[k + 1] < t_f[i] and k < m:  # find first k, that knots[k+1]>t_f[i]
                                    k += 1
                                hk_m = t_f[i] - knots[k]
                                hk_p = knots[k + 1] - t_f[i]
                                Value_f = hk_m / h[k] * g[k + 1] \
                                          + hk_p / h[k] * g[k] \
                                          - hk_m * hk_p * (h[k] + hk_m) / (6 * h[k]) * g2[k + 1] \
                                          - hk_m * hk_p * (h[k] + hk_p) / (6 * h[k]) * g2[k]
                                S += weights_f[i] * ((values_f[i] - np.exp(Value_f)) ** 2)

                        if t_df is not None and len(t_df) > 0:
                            k = 0
                            for i in range(ndf):
                                while knots[k] <= t_df[i] and knots[k + 1] < t_df[i] and k < m:  # find first k, that knots[k+1]>t_df[i]
                                    k += 1
                                hk_m = t_df[i] - knots[k]
                                hk_p = knots[k + 1] - t_df[i]
                                Value_f = hk_m / h[k] * g[k + 1] \
                                          + hk_p / h[k] * g[k] \
                                          - hk_m * hk_p * (h[k] + hk_m) / (6 * h[k]) * g2[k + 1] \
                                          - hk_m * hk_p * (h[k] + hk_p) / (6 * h[k]) * g2[k]
                                Value_df = (g[k + 1] - g[k]) / h[k] \
                                           - (h[k] / 6 - hk_m ** 2 / (2 * h[k])) * g2[k + 1] \
                                           + (h[k] / 6 - hk_p ** 2 / (2 * h[k])) * g2[k]
                                S += coef_df * weights_df[i] * ((values_df[i] - np.exp(Value_f) * Value_df) ** 2)

                        if t_d2f is not None and len(t_d2f) > 0:
                            k = 0
                            for i in range(nd2f):
                                while knots[k] <= t_d2f[i] and knots[k + 1] < t_d2f[i] and k < m:  # find first k, that knots[k+1]>t_d2f[i]
                                    k += 1
                                hk_m = t_d2f[i] - knots[k]
                                hk_p = knots[k + 1] - t_d2f[i]
                                Value_f = hk_m / h[k] * g[k + 1] \
                                          + hk_p / h[k] * g[k] \
                                          - hk_m * hk_p * (h[k] + hk_m) / (6 * h[k]) * g2[k + 1] \
                                          - hk_m * hk_p * (h[k] + hk_p) / (6 * h[k]) * g2[k]
                                Value_df = (g[k + 1] - g[k]) / h[k] \
                                           - (h[k] / 6 - hk_m ** 2 / (2 * h[k])) * g2[k + 1] \
                                           + (h[k] / 6 - hk_p ** 2 / (2 * h[k])) * g2[k]
                                Value_d2f = hk_m / h[k] * g2[k + 1] \
                                            + hk_p / h[k] * g2[k]
                                S += coef_d2f * weights_d2f[i] * ((values_d2f[i] - np.exp(Value_f) * (Value_df ** 2 + Value_d2f)) ** 2)

                        return S

                    # print("exp")
                    res = optimize.minimize(loss, np.zeros(m) ) #, options = {'maxiter':10000 , 'gtol' : 1e-5} )
                    # print(res)
                    # if not res.success:
                    #     print(res.message)
                    #     print("fail")
                    #     print(res)
                    #     return([np.nan])
                    g = res.x

                case _:
                    assert method in ["Lemke", "Lemke_njit", "cvxopt",
                                      "exp"], 'incorrect method specified, only "Lemke","Lemke_njit","cvxopt","exp" methods are possible'
            if g[0] == np.nan:
                print("g is np.nan")
                # print("alpha = ", alpha)
                return np.array([np.nan])
            g = g.reshape((m, 1))
        else:
            g = np.linalg.solve(D, c)
            need_repeat = False

        gamma = Rm1QT @ g  # After that spline is completely defined via g and gamma

        # Second derivative on the edges was zero
        g2 = np.append([0], gamma)
        g2 = np.append(g2, 0)

        if not All_Positive or not (add_knots or add_condition_without_knots) or max_added_knots <= 0 or method=="exp":
            need_repeat = False
        else:
            insert_indexes = []
            new_knots = []
            if add_condition_without_knots:
                # if 'Rm1QT' not in locals():
                #     Rm1QT = inv_R @ t_Q
                # if Z is None:
                # if 'Z' not in locals():
                #     Z = []
                num_new_conditions = 0

            for k in range(m-1):
                dg = (g[k+1] - g[k]) / h[k] - h[k] * (g2[k + 1] + 2 * g2[k]) / 6
                dddg = (g2[k+1] - g2[k]) / h[k]
                discr = g2[k]**2 - 2*dg*dddg
                if discr<0:
                    continue
                dt1 = (-g2[k] - np.sqrt(discr)) / (dddg)
                dt2 = (-g2[k] + np.sqrt(discr)) / (dddg)
                if dt1 > dt2:
                    dt1,dt2 = dt2,dt1
                if dt1>0 and dt1<h[k]:
                    hk_m = dt1
                    hk_p = h[k] - dt1
                    v = (hk_m * g[k + 1] + hk_p * g[k]) / h[k] - 1 / 6 * hk_m * hk_p * (
                            g2[k + 1] * (1 + hk_m / h[k]) + g2[k] * (1 + hk_p / h[k]))
                    if v <  -new_knots_tol:
                        if add_condition_without_knots:
                            V_cond = np.zeros(m)
                            P_cond = np.zeros(m)
                            V_cond[k] = hk_p / h[k]
                            V_cond[k+1] = hk_m / h[k]
                            P_cond[k] =  hk_m * hk_p*(h[k] + hk_p)/(h[k]*6)
                            P_cond[k+1] =  hk_m * hk_p*(h[k] + hk_m)/(h[k]*6)
                            P_cond = P_cond[1:m-1]
                            Zi = V_cond - P_cond @ Rm1QT
                            # print("g(sk+dt1)=", Zi @ g)
                            # print("v =", v)

                            Z.append(Zi)
                            num_new_conditions += 1
                            # print("Added condition for dt1 = ", dt1, " at position ",knots[k]+dt1, " where v = ", v)
                            new_knots_global.append((knots[k]+dt1).item())
                            # insert_indexes.append(k)
                        else:
                            insert_indexes.append(k+1)
                            new_knots.append((knots[k]+dt1).item())
                            new_knots_global.append((knots[k] + dt1).item())
                            # print("Added knot for dt1 = ", dt1, " at position ", knots[k] + dt1, " where v = ", v)

                if dt2>0 and dt2<h[k]:
                    hk_m = dt2
                    hk_p = h[k] - dt2
                    v = (hk_m * g[k + 1] + hk_p * g[k]) / h[k] - 1 / 6 * hk_m * hk_p * (
                            g2[k + 1] * (1 + hk_m / h[k]) + g2[k] * (1 + hk_p / h[k]))
                    if v <  -new_knots_tol:
                        if add_condition_without_knots:
                            V_cond = np.zeros(m)
                            P_cond = np.zeros(m)
                            V_cond[k] = hk_p / h[k]
                            V_cond[k+1] = hk_m / h[k]
                            P_cond[k] = (hk_m * hk_p*(h[k] + hk_p)/(h[k]*6))
                            P_cond[k+1] = (hk_m * hk_p*(h[k] + hk_m)/(h[k]*6))
                            P_cond = P_cond[1:-1]
                            Zi = V_cond - P_cond @ Rm1QT
                            # print("g(sk+dt2)=", Zi @ g)
                            # print("v =", v)

                            Z.append(Zi)
                            num_new_conditions += 1
                            # print("Added condition for dt2 = ", dt2, " at position ", knots[k] + dt2, " where v = ", v)
                            new_knots_global.append((knots[k] + dt2).item())
                            # insert_indexes.append(k)
                        else:
                            insert_indexes.append(k+1)
                            new_knots.append((knots[k]+dt2).item())
                            new_knots_global.append((knots[k] + dt2).item())
                            # print("Added knot for dt2 = ", dt2, " at position ", knots[k] + dt2, " where v = ", v)

            # print("new_knots = ", new_knots)
            # print("insert_indexes = ", insert_indexes)
            if len(new_knots)>0:
                # print(knots)
                knots = np.insert(knots,insert_indexes,new_knots)
                knots_number += len(new_knots)
                # print(knots)
            if len(new_knots)==0 and (not add_condition_without_knots or num_new_conditions==0):
                need_repeat = False
            max_added_knots -= 1
    # end While add_knots ==True

    # ===== Calculating and returning spline values at x coordinates  =====
    # print("g = ", g.flatten())
    # print("M @ g =", M @ g)

    if x is None:
        x = np.append(np.arange(knots[0], knots[-1], 1), knots[-1])
    if "y" in output:
        y = np.zeros(len(x))
    if "dy" in output:
        dy = np.zeros(len(x))
    if "d2y" in output:
        d2y = np.zeros(len(x))

    k = 0
    for j in range(len(x)):
        while x[j] > knots[k] + h[k] and k < m:
            k += 1
        hk_m = x[j] - knots[k]
        hk_p = knots[k + 1] - x[j]
        hk = h[k]
        if "y" in output:
            y[j] = (hk_m * g[k + 1] + hk_p * g[k]) / hk - 1 / 6 * hk_m * hk_p * (
                g2[k + 1] * (1 + hk_m / hk) + g2[k] * (1 + hk_p / hk))
        if "dy" in output:
            dy[j] = ((g[k + 1] - g[k]) / hk - ( hk/6 - hk_m**2/(2*hk))*g2[k+1] + (hk/6 - hk_p**2/(2*hk))*g2[k])
        if "d2y" in output:
            d2y[j] = g2[k]*hk_p/hk + g2[k+1]*hk_m/h[k]

    if "y" in output and method in ["exp"]:
        y = np.exp(y)

    if "integral" in output:    # return integral function F(t)
        integral = np.zeros(len(x))
        L = 0
        l = -1
        SumL = 0
        for j in range(len(x)):
            while x[j] > knots[L] + h[L] and L < m:
                L += 1
            while l < L:
                l += 1
                SumL += h[l] * (g[l + 1] + g[l]) / 2 - h[l] ** 3 * (g2[l + 1] + g2[l]) / 24
            hL_m = x[j] - knots[L]
            hL_p = knots[L + 1] - x[j]
            integral[j] = SumL - ((h[L] ** 2 - hL_m ** 2) * g[L + 1] + hL_p ** 2 * g[L]) / h[L] / 2 \
                               + hL_p ** 2 * ((hL_m + h[L]) ** 2 * g2[L + 1] - (hL_p ** 2 - 2 * h[L] ** 2) * g2[L]) / h[L] / 24

    if info:
        error_total = 0
        error_f = 0
        error_df = 0
        error_d2f = 0
        error_int = 0
        relative_sqr_error_f = 0
        relative_sqr_error_df = 0
        relative_sqr_error_d2f = 0
        relative_sqr_error_int = 0
        relative_abs_error_f = 0
        relative_abs_error_df = 0
        relative_abs_error_d2f = 0
        relative_abs_error_int = 0

        if t_f is not None and len(t_f) > 0:
            V = values_f - Cf @ g
            error_f = (np.transpose(V) @ Wf @ V).item()
            if np.any(values_f == 0):
                relative_abs_error_f = np.inf
                relative_sqr_error_f = np.inf
            else:
                V = np.abs(V / values_f)
                relative_abs_error_f = np.sum(V * weights_f) / nf  # (np.transpose(V) @ Wf @ V).item() / nf
                V = V ** 2
                relative_sqr_error_f = np.sqrt((np.transpose(V) @ Wf @ V).item() / nf)
        if t_df is not None and len(t_df) > 0:
            V = values_df - Cdf @ g
            error_df = (np.transpose(V) @ Wdf @ V).item()
            if np.any(values_df == 0):
                relative_abs_error_df = np.inf
                relative_sqr_error_df = np.inf
            else:
                V = np.abs(V / values_df)
                relative_abs_error_df = np.sum(V * weights_df) / ndf  # (np.transpose(V) @ Wdf @ V).item() / ndf
                V = V ** 2
                relative_sqr_error_df = np.sqrt((np.transpose(V) @ Wdf @ V).item() / ndf)
        if t_d2f is not None and len(t_d2f):
            V = values_d2f - Cd2f @ g
            error_d2f = (np.transpose(V) @ Wd2f @ V).item()
            if np.any(values_d2f == 0):
                relative_abs_error_d2f = np.inf
                relative_sqr_error_d2f = np.inf
            else:
                V = np.abs(V / values_d2f)
                relative_abs_error_d2f = np.sum(V * weights_d2f) / nd2f  # (np.transpose(V) @ Wd2f @ V).item() / nd2f
                V = V ** 2
                relative_sqr_error_d2f = np.sqrt((np.transpose(V) @ Wd2f @ V).item() / nd2f)
        if t_int_a is not None and len(t_int_a) > 0:
            V = values_int - Cint @ g
            error_int = (np.transpose(V) @ Wint @ V).item()
            if np.any(values_int == 0):
                relative_abs_error_int = np.inf
                relative_sqr_error_int = np.inf
            else:
                V = np.abs(V / values_int)
                relative_abs_error_int = np.sum(V * weights_int) / nint  # (np.transpose(V) @ Wint @ V).item() / nint
                V = V ** 2
                relative_sqr_error_int = np.sqrt((np.transpose(V) @ Wint @ V).item() / nint)

        error_penalty = (np.transpose(g) @ K @ g).item()

        error_total = error_f + coef_df * error_df + coef_d2f * error_d2f + coef_int * error_int + alpha * error_penalty
        fraction_error_f = error_f / error_total
        fraction_error_df = coef_df * error_df / error_total
        fraction_error_d2f = coef_d2f * error_d2f / error_total
        fraction_error_int = coef_int * error_int / error_total
        fraction_penalty = alpha * error_penalty / error_total
        result = {'x': x,
                # 'y': y,
                # 'dy':dy,
                # 'dy2': dy2,
                'g': g.flatten(), \
                'gamma': g2.flatten(), \
                'knots': knots, \
                'new_knots':new_knots_global, \
                'error_total': error_total, \
                'error_f': error_f, \
                'error_df': error_df, \
                'error_d2f': error_d2f, \
                'error_int,': error_int, \
                'error_penalty': error_penalty, \
                'fraction_error_f': fraction_error_f, \
                'fraction_error_df': fraction_error_df, \
                'fraction_error_d2f': fraction_error_d2f, \
                'fraction_error_int': fraction_error_int, \
                'fraction_penalty': fraction_penalty, \
                'relative_sqr_error_f': relative_sqr_error_f, \
                'relative_sqr_error_df': relative_sqr_error_df, \
                'relative_sqr_error_d2f': relative_sqr_error_d2f, \
                'relative_sqr_error_int': relative_sqr_error_int, \
                'relative_abs_error_f': relative_abs_error_f, \
                'relative_abs_error_df': relative_abs_error_df, \
                'relative_abs_error_d2f': relative_abs_error_d2f, \
                'relative_abs_error_int': relative_abs_error_int}
        if "y" in output:
            result["y"] = y
        if "dy" in output:
            result["dy"] = dy
        if "d2y" in output:
            result["d2y"] = d2y
        if "integral" in output:
            result["integral"] = integral
        return result
    else:
        if "y" in output:
            return y
        if "dy" in output:
            return dy
        if "d2y" in output:
            return d2y
        if "integral" in output:
            return integral






