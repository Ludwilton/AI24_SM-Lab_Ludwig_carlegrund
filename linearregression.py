import numpy as np
import scipy.stats as stats

class LinearRegression:
    def __init__(self, X, Y):
        self.Y = Y
        self.X = np.column_stack([np.ones(Y.shape[0]), X])
        self.n = Y.shape[0]
        self.b = self.fit()
        self.k = len(self.b) - 1
        self.SSE = self.calc_SSE()
        self.var = self.calculate_variance()
        self.S = np.sqrt(self.var)
    
    # implementera: 
    # property d that contains the number of features/parameters/dimensions of the model.
    # property n that contains the size of the sample.
    # property confidence_level that stores the selected confidence level.


     # inte en bra idé(?)
    def fit(self): # ols, returns coeffs as an array
        return np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y  
    
    def calc_SSE(self): # sum of squared errors, total deviation, predicted from actual, returns SSE
        return np.sum(np.square(self.Y-(self.X@self.b )))

    def calculate_variance(self):
        var = self.SSE/(self.n-self.k-1)
        return var


    def calculate_standard_deviation(self):
        sd = np.sqrt(self.var)
        return sd


    def regression_significance(self):
        Syy = (self.n*np.sum(np.square(self.Y)) - np.square(np.sum(self.Y))) / self.n
        SSR = Syy - self.SSE
        sig_statistic = (SSR/self.k) / self.S
        p_significance = stats.f.sf(sig_statistic,self.k , self.n - self.k - 1)
        return p_significance

        # Calculate the significance of the regression (f statistic and p value)


    def regression_relevance(self):# r^2, "proportion of the variance in the dependent variable that is predictable from the independent variable(s)"
        Syy = np.sum(np.square((self.Y - np.mean(self.Y))))
        SSR = Syy - self.SSE # make class variable
        Rsq = SSR / Syy
        return Rsq

    
    def variable_significance(self):
        # Calculate the significance of each variable
        # \hat\beta_i \pm t_{\alpha/2}\hat\sigma^2\sqrt{c_{ii}}

        # t_{sigma/2} is the appropriate point based on the T_{n-d-1} distribution and a confidence level {alpha}
        pass


    def pearson_correlation(self):
        # Calculate the Pearson correlation number "r"
        # R = \frac{X_a, X_b}{sqrt{VarX_a * VarX_b}}
        r = stats.pearsonr(self.X, self.Y)
        return r


    def confidence_interval(self):
        # Calculate the confidence interval
        pass


    