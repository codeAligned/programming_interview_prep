import quandl
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


quandl.ApiConfig.api_key =  '-UVzwX5xnzsm6x68A1Qq'

def fetch_data_quandl(code, name):
    """
    This provides a much cleaner output than just calling quandl
    (easier for merging later..)
    
    Parameters
    ----------
    code - Quandle code, e.g. "EIA/PET_A103600001_M"
    name - name for the column, e.g. "total"

    Returns
    -------
    dataframe
    
    """
    return quandl.get(code).rename(columns={'Value':name})

def fetch_all_data():
    """
    getch gdp and gas data
    Returns
    -------
    gdp, gas_data (pd.DataFrames)
    """
    gdp = fetch_data_quandl("FRED/GDP", 'gdp')

    gas_sales = []
    gas_sales.append(fetch_data_quandl("EIA/PET_A103600001_M",'total'))

    gas_sales.append(fetch_data_quandl("EIA/PET_A123600001_M",'regular'))
    gas_sales.append(fetch_data_quandl("EIA/PET_A143600001_M",'midgrade'))
    gas_sales.append(fetch_data_quandl("EIA/PET_A133600001_M",'premium'))

    gas_sales.append(fetch_data_quandl("EIA/PET_A163600001_M",'conventional'))
    gas_sales.append(fetch_data_quandl("EIA/PET_A023600001_M",'oxygenated'))
    gas_sales.append(fetch_data_quandl("EIA/PET_A013600001_M",'reformulated'))

    gas_sales_df = pd.concat(gas_sales,axis=1)
    return gdp, gas_sales_df


def merge_data(gdp, gas):
    """
    Merge GDP and gas data
    Note: we are resample gas data on a quarterly basis (and taking a sum of monthly figures)
    
    Parameters
    ----------
    gdp - pd.DataFrame
    gas - pd.DataFrame

    Returns
    -------
    gdp_gas - pd.DataFrame
    
    """
    gdp_gas = pd.merge(gdp.reset_index(), gas.resample('QS-JAN').sum().reset_index(), on='Date')



plt.interactive(False)
gdp_gas_no_dates = gdp_gas.iloc[:,1:]
gdp_gas_no_dates.head()
gdp_gas_no_dates.plot()
gdp_gas_no_dates[['gdp','total']].plot()
plt.show()

# try qtr over qtr change

changes = gdp_gas_no_dates.pct_change()
scatter_matrix(changes)
plt.show()
# test if things check out
df['total_1'] = df.iloc[:,1:4].sum(axis=1)
df['total_2'] = df.iloc[:,4:7].sum(axis=1)
df.drop(['total_1','total_2'],axis=1,inplace=True)

gdp = quandl.get("FRED/GDP")


gas_sales.append(quandl.get("EIA/PET_A103600001_M").rename(columns={'Value':'total'}))

gas_sales['regular'] = quandl.get("EIA/PET_A123600001_M")
gas_sales['midgrade'] = quandl.get("EIA/PET_A143600001_M")
gas_sales['premium'] = quandl.get("EIA/PET_A133600001_M")

gas_sales['conventional'] = quandl.get("EIA/PET_A163600001_M")
gas_sales['oxygenated'] = quandl.get("EIA/PET_A023600001_M")
gas_sales['reformulated'] = quandl.get("EIA/PET_A013600001_M")


df = pd.concat(gas_sales,axis=1)

gdp_gas = pd.merge(gdp.reset_index(), df.resample('QS-JAN').sum().reset_index(), on='Date')
gdp_gas.columns =

pd.DataFrame.from_dict(gas_sales, orient='columns')

pd.DataFrame(gas_sales.values())

pd.merge(gdp, gas_sales_df)


gas_sales_df.columns
gas_sales_df

gas_sales_df.tail()

gas_sales['midgrade'].tail()