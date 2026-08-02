import numpy as np
       
def rolling_vol(returns, h=21):
    '''Return Rolling Volatility'''
    return returns.rolling(h).std()



