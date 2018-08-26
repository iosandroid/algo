import sys
import pandas as pd

def cprintf(df):
    if not isinstance(df, pd.DataFrame):
        try:
            df = df.to_frame()
        except:
            raise ValueError('Object cannot be coerced to df.')
    
    print('-'*79)
    print('Data frame information')
    print('-'*79)
    print(df.tail(5))
    print('-'*50)
    print(df.info())
    print('-'*79)    