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
from MyLemke2 import *

def FunctionalSmoothingSpline(
			t_f = None,           # array of observation moments
			values_f = None,      # array of observation values
			weights_f = None,     # array of  weights
			t_df = None,          # array of first derivative moments
			values_df = None,     # array of first derivative values
			weights_df = None,    # array of first derivative weights
			coef_df = 1,          # coefficient of first derivative sum of squares
			t_d2f = None,         # array of second derivative moments
			values_d2f = None,    # array of second derivative values
			weights_d2f = None,   # array of second derivative weights
			coef_d2f = 1,         # coefficient of second derivative sum of squares
			t_int_a = None,       # array of interals start moments
			t_int_b = None,       # array of interals end moments
			values_int = None,    # array of interal values
			weights_int = None,   # array of interal weights
			coef_int = 1,         # coefficient of integral sum of squares
			knots = None,         # knots
			knots_number = None,  # number of knots
			alpha = 1,            # smoothing parameter
			x = None,             # output moments
            All_Positive = False,  # solve as monotone spline by Lemke algorithm
            integral = False,      # return integral function F(t)
			info = False):         # need info?

	
    if knots_number is None and knots is None:
        knots_number = 0
        if t_f is not None and len(t_f)>0:
            knots_number = len(t_f)
        if t_df is not None and len(t_df)>0 and knots_number<len(t_df):
            knots_number = len(t_df)
        if t_d2f is not None and len(t_d2f)>0 and knots_number<len(t_d2f):
            knots_number = len(t_d2f)
        if t_int_a is not None and len(t_int_a)>0 and knots_number<len(t_int_a):
            knots_number = len(t_int_a)

    assert knots_number >= 2, 'knots_number or observations should not be less than 2'
	
    m = knots_number # for short  
	
	# in case knots is not defined
    if knots is None or m != len(knots): # when knots_number defined, but knots not defined
        start_knot = + np.inf
        end_knot = - np.inf
        if t_f is not None and len(t_f)>0 :
            if start_knot > np.min(t_f):
                start_knot = np.min(t_f)
            if end_knot < np.max(t_f):
                end_knot = np.max(t_f)
        if t_df is not None and len(t_df)>0:
            if start_knot > np.min(t_df):
                start_knot = np.min(t_df)
            if end_knot < np.max(t_df):
                end_knot = np.max(t_df)
        if t_d2f is not None and len(t_d2f)>0:
            if start_knot> np.min(t_d2f):
                start_knot = np.min(t_d2f)
            if end_knot < np.max(t_d2f):
                end_knot =  max(t_d2f)
        if t_int_a is not None and len(t_int_a)>0 and len(t_int_b)>0:
            if start_knot > np.min(t_int_a):
                start_knot = np.min(t_int_a)    
            if end_knot < np.max(t_int_b):
                end_knot = np.max(t_int_b)     
        knots = np.linspace(start_knot,end_knot,m)

    h = np.zeros(m-1) #array of distance between knots
    h[0:(m - 1)] = knots[1:m] - knots[0:(m - 1)]
	
    #Matrix Q
    Q=np.zeros( (m, m-2) )
    for i in range(m - 2):
        Q[i,i] = 1/h[i];
        Q[i + 1,i] = - 1/h[i] - 1/h[i + 1];
        Q[i + 2,i] = 1/h[i + 1]
	
    #Matrix R
    R = np.zeros((m-2,m-2))
    for i in range(m - 2):
        R[i,i] = (h[i] + h[i + 1]) / 3;
        if (i < m-2 -1):
            R[i + 1,i] = h[i + 1]/6;
            R[i,i + 1] = h[i + 1]/6;
	
    #Matrix K calculation
    inv_R = np.linalg.inv(R)
    t_Q = np.transpose(Q)
    K = Q @ inv_R @ t_Q
	
    # ========= 1. Observation (t_f, values_f)  ===========
	
    if t_f is not None and len(t_f)>0 :
        nf = len(t_f) #number of observation coordinates
        assert len(values_f) == nf, 'length of values_f and t_f must be same'
        
        if weights_f is None:
            weights_f = np.ones(nf)
        assert len(weights_f) == nf, 'length of weights_f and t_f must be same'
        Wf = np.diag(weights_f) 
		
        #reorder observations (t_f, values_f) by appear time t_f
        ord = np.argsort(t_f)
        t_f = t_f[ord]
        values_f = values_f[ord]
        values_f = values_f.reshape((nf,1))
        
        #Filling in Vf and Pf matrices
        Vf = np.zeros((nf,m))
        Pf = np.zeros((nf,m))
        k = 0 # start knot
        for i in range(nf):
            while knots[k]<=t_f[i] and knots[k+1]<t_f[i] and k<knots_number: #find first k, that knots[k+1]>t_f[i]
                k += 1
            hk_m = t_f[i] - knots[k]
            hk_p = knots[k+1] - t_f[i]
            Vf[i,k] = hk_p/h[k] 
            Vf[i,k + 1] = hk_m/h[k]    
            Pf[i,k] = hk_m*hk_p*(h[k] + hk_p)/(6*h[k])   
            Pf[i,k + 1] = hk_m*hk_p*(h[k] + hk_m)/(6*h[k])
        Pf = Pf[:,1:(m - 1)] #don't need first and last column 
		
        #Matrix Cf calculation		
        Cf = Vf - Pf @ inv_R @ t_Q
        t_Cf = np.transpose(Cf)	
	
    # ========= 2. Observation (t_df, values_df)  ===========  
	
    if t_df is not None and len(t_df)>0:	
        ndf = len(t_df) #number of observation  
        assert len(values_df) == ndf, 'length of values_df and t_df must be same'
        
        if weights_df is None:
            weights_df = np.ones(ndf)
        assert len(weights_df) == ndf, 'length of weights_df and t_df must be same'
        Wdf = np.diag(weights_df) 
		
        ord = np.argsort(t_df)   #reorder observations (t_df, values_df)  by appear time t_df
        t_df = t_df[ord]
        values_df = values_df[ord]
        values_df = values_df.reshape((ndf,1))
        
        #Filling in Vdf and Pdf matrices
        Vdf = np.zeros((ndf, m))
        Pdf = np.zeros((ndf, m))
        k = 0 # start knot
        for i in range(ndf):
            while knots[k]<=t_df[i] and knots[k+1]<t_df[i] and k<m : #find first k, that knots[k + 1]>t_df[i]
                k = k + 1
            hk_m = t_df[i] - knots[k]
            hk_p = knots[k + 1] - t_df[i]
			
            Vdf[i,k] = - 1/h[k] 
            Vdf[i,k + 1] = 1/h[k]    
            Pdf[i,k] = - h[k]/6+(hk_p)**2/(2*h[k])   
            Pdf[i,k + 1] = h[k]/6-(hk_m)**2/(2*h[k])
        Pdf = Pdf[:,1:(m-1)] #don't need first and last column 
		
        #Matrix Cdf calculation
        Cdf = Vdf - Pdf @ inv_R @ t_Q
        t_Cdf = np.transpose(Cdf)
		
    # ========= 3. Observation (t_d2f, values_d2f)  ===========  
	
    if t_d2f is not None and len(t_d2f)>0:
        nd2f = len(t_d2f) #number of observation 
        assert len(values_d2f) == nd2f, 'length of values_d2f and t_d2f must be same'
        
        if weights_d2f is None:
            weights_d2f = np.ones(nd2f)
        assert len(weights_d2f) == nd2f, 'length of weights_d2f and t_d2f must be same'
        Wd2f = np.diag(weights_d2f) 	
        #reorder observations (t_d2f, values_d2f)  by appear time t_d2f
        ord = np.argsort(t_d2f)
        t_d2f = t_d2f[ord]
        values_d2f = values_d2f[ord]
        values_d2f = values_d2f.reshape((nd2f,1))
        
        #Filling in Pd2f matrices
        Pd2f=np.zeros((nd2f,m))
        k = 0 # start knot
        for i in range(nd2f):
            while knots[k]<=t_d2f[i] and knots[k+1]<t_d2f[i] and k<m: #find first k, that knots[k+1]>t_d2f[i]
                k = k + 1
            hk_m = t_d2f[i] - knots[k]
            hk_p = knots[k + 1] - t_d2f[i]
			
            Pd2f[i,k] = - hk_p/h[k]  
            Pd2f[i,k+1] = - hk_m/h[k]
        Pd2f = Pd2f[:,1:(m - 1)] #don't need first and last column 
		
        #Matrix Cd2f calculation
        Cd2f = - Pd2f @ inv_R @ t_Q
        t_Cd2f = np.transpose(Cd2f)   
	
    # ========= 4. Observation (t_int_a, t_int_b, values_int)  ===========  
	
    if t_int_a is not None and len(t_int_a)>0:	
        nint=len(t_int_a) #number of observation
        assert len(t_int_b) == nint, 'length of t_int_b and t_int_a must be same'
        assert len(values_int) == nint, 'length of values_int and t_int_a must be same'
        if weights_int is None:
            weights_int = np.ones(nint)
        assert len(weights_int) == nint, 'length of weights_int and t_int_a must be same'
        Wint = np.diag(weights_int) 
		
        #reorder observations (t_int_a, t_int_b, values_int) by appear time t_int_a
        ord = np.argsort(t_int_a)  # order(t_int_a, t_int_b,values_int)
        t_int_a = t_int_a[ord]
        t_int_b = t_int_b[ord]
        values_int = values_int[ord]
        values_int = values_int.reshape((nint,1))
		
        #Filling in Vint and Pint matrices
        Vint = np.zeros((nint, m))
        Pint = np.zeros((nint, m))
        k = 0 # start knot
        for i in range(nint):
            while knots[k]<=t_int_a[i] and knots[k + 1]<t_int_a[i] and k<m: #find first k, that knots[k + 1]>t_int_a[i]
                k = k + 1      
			
            #finding L, it can be 0
            L = 0
            while t_int_b[i] > knots[k + L + 1] and k+L+1 < m-1:
                L += 1
	
            hk_m = t_int_a[i] - knots[k]
            hk_p = knots[k + 1] - t_int_a[i]
            hkL_m = t_int_b[i] - knots[k + L]
            hkL_p = knots[k + L + 1] - t_int_b[i]
			
            Vint[i,k] = (hk_p)**2/h[k]/2  
            Pint[i,k] = h[k]**3/24 - (hk_m)**2*(hk_p + h[k])**2/h[k]/24
            l = 1
            while l<=L:
                Vint[i, k + l] = (h[k + l - 1] + h[k + l])/2
                Pint[i, k + l] = (h[k + l - 1]**3 + h[k + l]**3)/24
                l += 1
            Vint[i, k + 1] = Vint[i, k + 1] - (hk_m)**2/h[k]/2
            Pint[i, k + 1] = Pint[i, k + 1] + (hk_m)**2*((hk_m)**2 - 2*h[k]**2)/h[k]/24
            Vint[i, k + L] = Vint[i, k + L] - (hkL_p)**2/h[k + L]/2
            Pint[i, k + L] = Pint[i, k + L] + (hkL_p)**2*((hkL_p)**2 - 2*h[k + L]**2)/h[k + L]/24    
            Vint[i, k + L + 1] = (hkL_m)**2/h[k + L]/2
            Pint[i, k + L + 1] = h[k + L]**3/24 - (hkL_p)**2*(hkL_m + h[k + L])**2/h[k + L]/24
			
        Pint=Pint[:,1:(m - 1)] #don't need first and last column
		
        #Matrix Cint calculation		
        Cint = Vint - Pint @ inv_R @ t_Q
        t_Cint = np.transpose(Cint)
		     		
	
    # ============ Calculation =============
	
    # matrix A
    A = alpha * K
    if t_f is not None and len(t_f)>0:
        A = A + t_Cf @ Wf @ Cf
    if t_df is not None and len(t_df)>0:
        A = A + coef_df * (t_Cdf @ Wdf @ Cdf)  
    if t_d2f is not None and len(t_d2f)>0:
        A = A + coef_d2f * t_Cd2f @ Wd2f @ Cd2f  
    if t_int_a is not None and len(t_int_a)>0:
        A = A + coef_int * t_Cint @ Wint @ Cint 
	
    # matrix D
    D = np.zeros( (m, 1))
    if t_f is not None and len(t_f)>0:
        D = D + t_Cf @ Wf @ values_f
    if t_df is not None and len(t_df)>0:
        D = D + coef_df * t_Cdf @ Wdf @ values_df
    if t_d2f is not None and len(t_d2f)>0:
        D = D + coef_d2f * t_Cd2f @ Wd2f @ values_d2f
    if t_int_a is not None and len(t_int_a)>0:
        D = D + coef_int * t_Cint @ Wint @ values_int
        
    #Calculation of g and gamma
    if All_Positive:
        g, exit_code, exit_string = Lemke_njit(A, -D, maxIter = 10000) # if you dont want njit optimization with numba, just delete _njit and call Lemke(A, -D, maxIter = 10000)
        if g[0] == np.nan:
            print("g is np.nan")
            print("alpha = ", alpha)
            return np.array([np.nan])
        g = g.reshape((m,1))
    else:
        g = np.linalg.solve(A , D)
    gamma = inv_R @ t_Q @ g   #After that spline is completely defined via g and gamma
	
	
    # ===== Calculating and returning spline values at x coordinates  =====
	
    #Second derivative on the edges was zero
    g2 = np.append([0],gamma)
    g2 = np.append(g2,0)
	
    if x is None:
        x = np.append( np.arange(knots[0],knots[-1],1) , knots[-1])
    y = np.zeros(len(x))
    
    if not integral:
        k = 0  
        for j in range(len(x)):
            while x[j]>knots[k]+h[k] and k<m:
                k += 1
            hk_m = x[j] - knots[k]
            hk_p = knots[k + 1] - x[j]
            y[j] = (hk_m*g[k + 1] + hk_p*g[k])/h[k] - 1/6*hk_m*hk_p*(g2[k + 1]*(1 + hk_m/h[k]) + g2[k]*(1 + hk_p/h[k])  )

    else: #return integral function F(t)
        L = 0 
        l = -1
        SumL = 0
        for j in range(len(x)):
            while x[j]>knots[L]+h[L] and L<m:
                L += 1
            while l < L:
                l += 1
                SumL += h[l]*(g[l+1]+g[l])/2 - h[l]**3*(g2[l+1] +g2[l])/24
            hL_m = x[j] - knots[L]
            hL_p = knots[L + 1] - x[j]
            y[j] = SumL - ( (h[L]**2-hL_m**2)*g[L+1] + hL_p**2*g[L] )/h[L]/2  \
                        + hL_p**2*((hL_m+h[L])**2*g2[L+1] - (hL_p**2-2*h[L]**2)*g2[L])/h[L]/24
        
    if info:
        error_total = 0
        error_f = 0
        error_df = 0
        error_d2f = 0
        error_int = 0    
        error_penalty = 0
        fraction_error_f = 0
        fractio_error_df = 0
        fractio_error_d2f = 0
        fractio_error_int = 0    
        fractio_penalty = 0    
        relative_sqr_error_f = 0
        relative_sqr_error_df = 0
        relative_sqr_error_d2f = 0
        relative_sqr_error_int = 0
        relative_abs_error_f = 0
        relative_abs_error_df = 0
        relative_abs_error_d2f = 0
        relative_abs_error_int = 0  
  
        if t_f is not None and len(t_f)>0:
            V = values_f - Cf @ g
            error_f = (np.transpose(V) @ Wf @ V).item()
            V = np.abs(V / values_f)
            relative_abs_error_f = (np.transpose(V) @ Wf @ V).item() / nf
            V = V**2
            relative_sqr_error_f = np.sqrt((np.transpose(V) @ Wf @ V).item() / nf)
        if t_df is not None and len(t_df)>0:
            V = values_df - Cdf @ g
            error_df = (np.transpose(V) @ Wdf @ V).item()
            V = np.abs(V / values_df) 
            relative_abs_error_df = (np.transpose(V) @ Wdf @ V).item() / ndf
            V = V**2
            relative_sqr_error_df = np.sqrt((np.transpose(V) @ Wdf @ V).item() / ndf) 
        if t_d2f is not None and len(t_d2f):
            V = values_d2f - Cd2f @ g
            error_d2f = (np.transpose(V) @ Wd2f @ V).item()
            V = np.abs(V / values_d2f) 
            relative_abs_error_d2f = (np.transpose(V) @ Wd2f @ V).item() / nd2f
            V = V**2
            relative_sqr_error_d2f = np.sqrt((np.transpose(V) @ Wd2f @ V).item() / nd2f)
        if t_int_a is not None and len(t_int_a)>0:
            V = values_int - Cint @ g
            error_int = (np.transpose(V) @ Wint @ V).item()
            V = np.abs(V / values_int) 
            relative_abs_error_int = (np.transpose(V) @ Wint @ V).item() / nint
            V = V**2
            relative_sqr_error_int = np.sqrt((np.transpose(V) @ Wint @ V).item() / nint)
            
        error_penalty = (np.transpose(g) @ K @ g).item()

        error_total = error_f + coef_df*error_df + coef_d2f*error_d2f + coef_int*error_int + alpha*error_penalty
        fraction_error_f = error_f/error_total
        fraction_error_df = coef_df*error_df/error_total
        fraction_error_d2f = coef_d2f*error_d2f/error_total
        fraction_error_int = coef_int*error_int/error_total    
        fraction_penalty = alpha*error_penalty/error_total  
		
        return	{   'x':x, 
                    'y':y, \
                'g':g, \
                'gamma':g2, \
                'knots':knots, \
                'error_total':error_total, \
                'error_f':error_f, \
                'error_df':error_df, \
                'error_d2f':error_d2f, \
                'error_int,':error_int,     \
                'error_penalty':error_penalty, \
                'fraction_error_f':fraction_error_f, \
                'fraction_error_df':fraction_error_df, \
                'fraction_error_d2f':fraction_error_d2f, \
                'fraction_error_int':fraction_error_int,     \
                'fraction_penalty':fraction_penalty,     \
                'relative_sqr_error_f':relative_sqr_error_f, \
                'relative_sqr_error_df':relative_sqr_error_df, \
                'relative_sqr_error_d2f':relative_sqr_error_d2f, \
                'relative_sqr_error_int':relative_sqr_error_int, \
                'relative_abs_error_f':relative_abs_error_f, \
                'relative_abs_error_df':relative_abs_error_df, \
                'relative_abs_error_d2f':relative_abs_error_d2f, \
                'relative_abs_error_int':relative_abs_error_int }
    else: 
        return y  




