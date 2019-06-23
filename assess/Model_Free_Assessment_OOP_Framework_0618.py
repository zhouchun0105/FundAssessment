
import pandas as pd
import numpy as np
import scipy.special
import scipy.optimize
from scipy.stats import norm


df = pd.read_csv("Sample_hedge_fund.csv")
df.dropna(0,how="all",inplace=True)
df["DATE"] = pd.to_datetime(df['DATE'])



class HF_Model_Free_Assessment():
    
    rf_df = pd.read_csv("monthly_rf_rate.csv")
    rf_df['DATE'] = pd.to_datetime(rf_df['DATE'])
    
    def __init__(self,date,performance,AUM):
        
        self.date = date
        self.performance = performance
        self.AUM = AUM
        
        source_date = pd.DataFrame(self.date)
        
        self.rf = source_date.merge(rf_df,how="left").GS1M
    
    
    
    #### Basic Function ####
    
    def show_all(self):
        df = pd.concat([self.date,self.performance,self.AUM],1)
        return(df)
    
    def compound_return(self):
        compound_return = np.cumprod(self.performance/100+1)
        return compound_return
    
    def excessive_downside_deviation(self,target):
        excessive_return = self.performance/100 - target
        excessive_return = excessive_return.dropna()
        negative_excessive_return = excessive_return[excessive_return <= 0]
        EDD = np.std(negative_excessive_return)
        return(EDD)
        
    
    #######################################
    
    
    #### Basic Assessment Function ####
    
    def Sharpe_ratio(self):
        print("This is used to calculate Sharpe Ratio")
        pass
    
    def Treynor_ratio(self):
        print("This is used to calculate Treynor Ratio")
        pass
    
    def Jarque_Bera(self):
        print("This is used to calculate Jarque Bera Statistics")
        pass
    
    
    #######################################
    
    
    #### Adjusted Sharpe Ratios Family ####
    
    def Asymmetry_Adj_sharpe_ratio(self,target_return):

        # Calculate the left-hand side of the function
        EDD = self.excessive_downside_deviation(target_return)
        std = np.std(self.performance.dropna()/100)
        var_ratio = (EDD**2)/(std**2)

        # find appropriate booundary of the solution

        def fun(lam):
            return (lam**2+1)*scipy.special.ndtr(-lam)-lam*norm.pdf(lam)-var_ratio

        a = -2
        b = 2
        counter = 1
        while np.sign(fun(a)) == np.sign(fun(b)):
            counter =+ 1
            a = a+0.1
            b = b-0.1
            if counter>1000:
                raise Exception('Encounter Numerical Mistake.')

        # Solve the optimization problem
        try:
            final_lam = scipy.optimize.brentq(fun,a,b)
            return(final_lam)
        except:
            print("Can not find unique Solution.")
            pass
    
    
    def Autocorrelation_Adj_sharpe_ratio(self):
        print("This is used to calculate Autocorrelation Adjusted Sharpe Ratio.")
    
    def VaR_Adj_sharpe_ratio(self):
        print("This is used to calculate Modified Sharpe Ratio.")
    
    
    #######################################
    
    #### Non-Sharpe Based Measurement ####
    
    def stutzer_index(self):
        print("This is used to calculate Modified Sharpe Ratio.")
    
    
    def omega(self):
        print("This is used to calculate Modified Sharpe Ratio.")
    
    
    


date = df.DATE
performance = df.Performance
AUM = df.AUM


sample_HF_assessment = HF_Model_Free_Assessment(date,performance,AUM)

