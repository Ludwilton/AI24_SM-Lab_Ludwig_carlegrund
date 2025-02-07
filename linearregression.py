import numpy as np
import scipy.stats as stats

class LinearRegression:
    def __init__(self, X, Y, column_names):
        self.X = np.column_stack([np.ones(Y.shape[0]), X])
        self.Y = Y
        self.column_names = column_names

    
    @property
    def d(self): # contains the number of features/parameters/dimensions of the model.
        return self.X.shape[1] - 1
    
    @property
    def n(self): #  contains the size of the sample
        return self.Y.shape[0]


    @property 
    def var(self): # calculates the variance
        return self.SSE/(self.n-self.d-1)


    @property
    def S(self): # calculates the standard deviation
        return np.sqrt(self.var)


    @property
    def confidence_level(self): 
        r2 = self.regression_relevance()
        return round(r2, 3)
        

    @property
    def SSE(self): # sum of square errors, total deviation, predicted from actual, 
        return np.sum(np.square(self.Y-(self.X@self.b )))
    

    @property
    def b(self): # ols, returns coeffs as an array, aka fit, regression etc
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y  
    

    @property
    def Syy(self): # total variance for y
        return (self.n*np.sum(np.square(self.Y)) - np.square(np.sum(self.Y))) / self.n


    @property
    def SSR(self): # sum of squared regression, returns SSR
        return self.Syy - self.SSE


    @property
    def c(self): # covariance matrix
        return np.linalg.pinv(self.X.T @ self.X)*self.var


    def regression_significance(self): # F-test, "is the model significant?"
        sig_statistic = (self.SSR/self.d) / self.var
        p_val = stats.f.sf(sig_statistic,self.d , self.n - self.d - 1)
        return p_val


    def regression_relevance(self):# r^2, "proportion of the variance in the dependent variable that is predictable from the independent variable(s)"
        Rsq = self.SSR / self.Syy
        return Rsq

    
    def variable_significance(self):
        p_values = []
        for i in range(1, self.d +1):
            b_statistic = self.b[i] / (self.S * np.sqrt(self.c[i,i]))
            p_b = 2*min(stats.t.cdf(b_statistic, self.n-self.d-1), stats.t.sf(b_statistic, self.n-self.d-1))
            p_values.append(float(p_b))
        return p_values


    def pearson_correlation(self):
        correlations = []
        for i in range(1, self.d + 1):
            for j in range(i+1, self.d + 1):
                pear, p_val = stats.pearsonr(self.X[:,i], self.X[:,j])
                correlations.append((self.column_names[i-1], self.column_names[j-1], float(pear)))
        return correlations


    def confidence_interval(self):
        alpha = 1 - self.confidence_level
        df = self.n - self.d - 1
        t_val = stats.t.ppf(1 - alpha/2, df)
        intervals = []
        for i, coef in enumerate(self.b):
            se = self.S * np.sqrt(self.c[i, i])
            lower = coef - t_val * se
            upper = coef + t_val * se
            pm = t_val * se
            intervals.append((coef,lower, upper, pm))
        return (intervals)


    def __str__(self):
        return f"""
samples: {self.n}
    
features: {self.d} ({self.column_names})

intercept: {self.b[0]}
    
coefficients: {[f"{name}: {value:.5f}" for name, value in zip(self.column_names, self.b[1:])]}
        
variance: {self.var:.5f}
        
standard deviation: {self.S:.5f}

regression significance: P:{self.regression_significance()} 
        
regression relevance / R^2 value: {self.regression_relevance():.5f}
        
The R^2 value puts the confidence value of the model to {self.confidence_level}.

pearson correlation: {self.pearson_correlation()}

variable significance: P: {[f"{name}: {value}" for name, value in zip(self.column_names, self.variable_significance())]}
        
confidence intervals: 
{[f"{name}: {coef:.5f} ± {pm:.5f}({lower:.5f}, {upper:.5f})" for name, (coef,lower, upper,pm) in zip(self.column_names, self.confidence_interval()[1:])]}
"""
