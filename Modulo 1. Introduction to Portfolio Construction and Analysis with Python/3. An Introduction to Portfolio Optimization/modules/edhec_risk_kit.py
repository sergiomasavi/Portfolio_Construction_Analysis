# Librerías
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import warnings
import numpy as np
import scipy.stats # skewness and kurtosis
from scipy.stats import norm

def get_ind_returns(filename):
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly
    Returns.
    
    Args:
    -----
    filename [{str}] -- Directorio del fichero csv.
    
    
    Returns:
    -----
    df_ind [{pandas.DataFrame}] -- DataFrame cargado con el csv con el formato
                                   adecuado.
    """
    df_ind = pd.read_csv(filename,
                         header=0,
                         index_col=0,
                         parse_dates=True)/100
    
    # Convertir indice a formato fecha
    df_ind.index = pd.to_datetime(df_ind.index, format='%Y%m').to_period('M')

    # Formatear nombre de columnas
    df_ind.columns = df_ind.columns.str.strip()
    
    return df_ind

def get_ffme_returns(directory, filename=None):
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    
    filename =f'{directory}/Portfolios_Formed_on_ME_monthly_EW.csv'
    
    me_m = pd.read_csv(filename,
                       header=0, index_col=0, na_values=-99.99)


    # Formato de la marca temporal
    me_m.index = pd.to_datetime(me_m.index, format="%Y%m")
    me_m.index = me_m.index.to_period('M')


    rets = me_m[['Lo 20', 'Hi 20']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    
    return rets

def get_hfi_returns(directory):
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    filename = directory + 'edhec-hedgefundindices.csv'
    hfi = pd.read_csv(filename,
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')

    return hfi

def drawdown(stock_returns: pd.Series, capital_inicial:float):
    """
    Toma una serie temporal de rendimientos de activos y devuelve un DataFrame
    con columnas para:
        - El índice de riqueza (wealth index).
        - Picos anteriores (previous peak).
        - El drowdown porcentual. 
        
    Args:
    ------
    stock_returns [{pandas.Series}] -- Serie de retornos de un activo.
    
    Returns:
    ------ 
    df_drawdowns [{pandas.DataFrame}] -- Dataframe de drawdowns cuyas columnas 
                                         corresponden con el índice de riqueza,
                                         los picos previos y el drawdown.
    """
    df_drawdowns = pd.DataFrame()
    
    # Calcular índice de riqueza = Crecimiendo de cada unidad monetaria del capital inicial
    df_drawdowns['Wealth_Index'] = capital_inicial*(1 + stock_returns).cumprod() 
    
    # Calcular picos previos
    df_drawdowns['Previous_Peaks'] = df_drawdowns['Wealth_Index'].cummax()
    
    # Calcular drowdown en valor porcentual
    df_drawdowns['Drawdown'] = (df_drawdowns['Wealth_Index'] - df_drawdowns['Previous_Peaks'])/df_drawdowns['Previous_Peaks']
    
    return df_drawdowns

def max_drawdown(drawdowns):
    """
    Calcular Max Drawdown a partir de la serie temporal de drawdowns para
    delvolver su valor y su marca temporal
    
    Args:
    -----
    drawdowns [{pandas.Series}] -- Drawdowns
    """
    # Calcular Max Drawdown
    max_drawdown = drawdowns.min()
    date_max_drawdown = drawdowns.idxmin()

    return max_drawdown, date_max_drawdown.strftime('%Y-%m')

def plot_drawdowns(df_drawdowns, title):
    """
    Visualización drawdowns con plotly. 
    
    Args:
    -----
    df_drawdowns [{pandas.DataFrame}] -- DataFrame de drawdowns resultados de 
                                         aplicar la función drawdowns(...)
    title [{str}] -- Título del gráfico
    
    Returns:
    ------
    None
    """
    fig = make_subplots(rows=2, cols=1,shared_xaxes=True)

    # Formatear columna de marca temporal
    df_drawdowns['date_time'] = df_drawdowns.index
    df_drawdowns['date_time'] = df_drawdowns['date_time'].apply(str)
    df_drawdowns.reset_index(drop=True, inplace=True)
    df_drawdowns['date_time'] = pd.to_datetime(df_drawdowns['date_time'], format='%Y-%m')
    df_drawdowns.set_index('date_time', inplace=True)

    fig.add_trace(
        go.Scatter(
            x=df_drawdowns.index, 
            y=df_drawdowns['Wealth_Index'],
            mode='lines',
            name='Wealth Index',
            line = dict(color='blue', width=1)
            
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df_drawdowns.index, 
            y=df_drawdowns['Previous_Peaks'],
            mode='lines',
            name='Previous Peaks',
            line = dict(color='green', width=2, dash='dot')
        ),
        row=1,
        col=1
    )
        
    fig.add_trace(
        go.Scatter(
            x=df_drawdowns.index, 
            y=df_drawdowns['Drawdown'],
            mode='lines',
            name='Drawdown',
            line = dict(color='red', width=1)
        ),
        row=2,
        col=1
    )
    
    # Calcular max drawdown
    date_max_drawdown = df_drawdowns["Drawdown"].idxmin()
    max_drawdown = df_drawdowns.loc[date_max_drawdown]['Drawdown']

    df_drawdowns['MaxDrawdown'] = np.nan
    df_drawdowns.loc[date_max_drawdown, 'MaxDrawdown'] = max_drawdown

    fig.add_trace(
        go.Scatter(
            x=df_drawdowns.index, 
            y=df_drawdowns['MaxDrawdown'],
            mode='markers',
            name='Max Drawdown',
            marker=dict(
                color='black',
                size=10,
                opacity=0.8,
                symbol='triangle-up'
            )),
        row=2,
        col=1
    )

    fig.update_layout(height=600, width=1000, title_text=title)
    fig.update_traces(hovertemplate="%{y:.5f}")

    fig.update_yaxes(tickprefix="")

    fig.update_xaxes(
        showspikes=True, spikethickness=2, spikecolor="#999999", spikemode="across"
    )

    fig.update_layout(
        hovermode="x unified",
        hoverdistance=200,
        spikedistance=200,
        transition_duration=500,
    )

    fig.show()

def skewness(r):
    """
    Alternativa a scipy.stats.skew()
    
    Calcula la asimetría de una serie temporal de rendimientos de entrada
    
    Args:
    -----
    r {[pandas.Series or pandas.DataFrame]} -- Serie temporal de rendimientos
    
    
    Returns:
    -----
    skewness {[float or Series]} -- Valor de asimetría
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0) # population standard deviation (ddof=0)
    exp_demeaned_r = (demeaned_r**3).mean()
    skewness = exp_demeaned_r/sigma_r**3
    return skewness

def kurtosis(r):
    """
    Alternativa a scipy.stats.skew()
    
    Calcula la kurtosis de una serie temporal de rendimientos de entrada
    
    Args:
    -----
    r {[pandas.Series or pandas.DataFrame]} -- Serie temporal de rendimientos
    
    
    Returns:
    -----
    kurtosis {[float or Series]} -- Valor de kurtosis
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0) # population standard deviation (ddof=0)
    exp_demeaned_r = (demeaned_r**4).mean()
    kurtosis = exp_demeaned_r/sigma_r**4
    return kurtosis

def is_normal(r, level=0.01):
    """
    Aplica el test de Jarque-Bera para determinar si una Serie es normal o no
    
    
    Args:
    ------
    r {[pandas.Series]} -- Serie temporal de rendimientos
    level {[float]} -- Nivel de confianza del test. La prueba se aplica por defecto al nivel del 1%


    Returns:
    ---------
    aceptar_hipotesis {[bool]} -- Devuelve Verdadero si se acepta la hipótesis de normalidad, Falso en caso contrario
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level
    
def negative_semideviation(r):
    """
    Devuelve la semidesviación en la semidesviación negativa de r
    
    Args:
    ------
    r [{pd.Series or pd.DataFrame}] -- Serie(s) temporal de retornos.
    
    Returns:
    ----------
    semideviation {[float or pd.Series]} -- Semidesviación negativa de los retornos
    """
    is_negative = r < 0
    ns = r[is_negative].std(ddof=0)
    return ns

def var_historic(r, level=5):
    """
    Devuelve el Valor en Riesgo utilizando el método de histórico en un nivel 
    especificado, es decir, devuelve el número tal que el porcentaje de 
    "nivel" de los rendimientos se sitúan por debajo de ese número, y 
    el porcentaje (de nivel 100) está por encima.
    
    Args:
    r [{pd.Series or pd.DataFrame}] -- Serie(s) temporal de retornos.
    level [{float}] -- Nivel de riesgo
    
    Returns:
    ----------
    var {[float or pd.Series]} -- VaR calculado con el método de históricos.
    """
    
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    
    elif isinstance(r, pd.Series):
        VaR = -np.percentile(r, level)
        return VaR
    
    else:
        raise TypeError('El tipado de la serie temporal de rendimientos debe ser pandas.Series o pandas.DataFrame')
     
def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
       
def var_gaussian(r, level=5):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    z_score =  -(r.mean() + z*r.std(ddof=0))
    return z_score

def var_cornish_fisher(r, level=5):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    # modify the Z score based on observed skewness and kurtosis
    s = skewness(r)
    k = kurtosis(r)
    z = (z +
            (z**2 - 1)*s/6 +
            (z**3 -3*z)*(k-3)/24 -
            (2*z**3 - 5*z)*(s**2)/36
        )
    z_cf = -(r.mean() + z*r.std(ddof=0))

    return z_cf

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def cvar_gaussian(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """     
        
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_gaussian(r, level=level)
        return -r[is_beyond].mean()
    
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_gaussian, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def cvar_cornish_fisher(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """     
        
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_cornish_fisher(r, level=level)
        return -r[is_beyond].mean()
    
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_cornish_fisher, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
       
def var_analysis(returns : pd.DataFrame, level):
    """
    
    
    
    Args:
    ------
    returns [{pandas.DataFrame}] -- DataFrame de retornos de distintos activos
    level [{float}] -- Nivel de VaR
    
    Returns:
    ------
    
    """
    df_result = pd.DataFrame(index=returns.columns)
    
    # Var Historic
    var_historic_ = var_historic(returns, level=level)
    df_var_historic = pd.DataFrame(var_historic_)
    df_var_historic.columns= ['Historic']
    df_result = df_result.join(df_var_historic)
    
    # VaR Gaussian
    var_gaussian_ = var_gaussian(returns, level=level)
    df_var_gaussian = pd.DataFrame(var_gaussian_)
    df_var_gaussian.columns= ['Gaussian']
    df_result = df_result.join(df_var_gaussian)
    
    # VaR Cornish-Fisher
    var_cornish_fisher_ = var_cornish_fisher(returns, level=level)
    df_var_cornish_fisher = pd.DataFrame(var_cornish_fisher_)
    df_var_cornish_fisher.columns= ['Cornish-Fisher']
    df_result = df_result.join(df_var_cornish_fisher)
    
    
    # Visualización
    layout = go.Layout(
    autosize=False,
    width=1150,
    height=550,

    xaxis= go.layout.XAxis(linecolor = 'black',
                           linewidth = 1,
                           mirror = True),

    yaxis= go.layout.YAxis(linecolor = 'black',
                           linewidth = 1,
                           mirror = True),

    margin=go.layout.Margin(l=50,
                            r=50,
                            b=100,
                            t=100,
                            pad = 4))
    
    fig = go.Figure(data=[
        go.Bar(name='Historic',       x=df_result.index, y=df_result['Historic']),
        go.Bar(name='Gaussian',       x=df_result.index, y=df_result['Gaussian']),
        go.Bar(name='Cornish-Fisher', x=df_result.index, y=df_result['Cornish-Fisher'])
    ],
                   layout=layout)
    
    # Change the bar mode
    fig.update_layout(barmode='group', title_text=f'Value at Risk at {level}%')
    fig.show()
    
    return df_result

def annualize_rets(r, periods_per_year):
    """
    Computar la anualización de un conjunto de rendimientos.
    
    Args:
    -----
    r [{}] -- 
    periods_per_year [{}] --
    
    
    Returns:
    -----
    
    
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    annualize_returns =  compounded_growth**(periods_per_year/n_periods) - 1
    return annualize_returns

def annualize_vol(r, periods_per_year):
    """
    Computar la volatilidad anualizada de un conjunto de rendimientos
    
    Args:
    -----
    r [{}] -- 
    periods_per_year [{}] --
    
    
    Returns:
    -----
    
    """
    annualize_volatility = r.std()*periods_per_year**0.5
    return annualize_volatility

def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    Computar el sharpe ratio anualizado de un conjunto de rendimientos.
    
    Args:
    -----
    
    r [{}] -- 
    periods_per_year [{}] --
    risk_free_rate [{float}] -- Tasa libre de riesgo.

    
    
    
    Returns:
    -----
    
    """
    
    # Convertir la tasa libre de riesgo
    rf_per_period = (1 + risk_free_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    
    return ann_ex_ret/ann_vol

def portfolio_return(weights, expected_returns):
    """
    Calcula el rendimiento de una cartera a partir de los rendimientos esperados
    y las ponderaciones que la componen.
    
    
    Args:
    ------
    weights [{numpy.matrix}] -- Matriz de pesos Nx1.
    expected_returns [{numpy.matrix}] -- Matriz de rendimientos anualizados esperados Nx1.
    
    
    Returns:
    ------
    portfolio_return [{float}] -- Rendimiento de la cartera ponderada.
    """
    
    portfolio_ret = weights.T @ expected_returns
    
    return portfolio_ret
     
def portfolio_volatility(weights, covmat):
    """
    Calcula la volatilidad de una cartera a partir de una matriz de covarianza
    y los pesos que la componen.
    
    Args:
    ------
    weights [{numpy.matrix}] -- Matriz de pesos Nx1.
    covmat [{numpy.matrix}] -- Matriz de covarianza esperados NxN.
    
    
    Returns:
    ------
    portfolio_vol [{float}] -- Volatilidad de la cartera ponderada.
    
    """
    portfolio_vol = (weights.T @ covmat @ weights)**0.5
    
    return portfolio_vol

def plot_ef2(n_points, expected_returns, covmat, title):
    """
    Visualización de la frontera eficiente basada en dos activos.
    
    
    Args:
    ------
    n_points [{int}] -- Número de puntos de la frontera eficiente
    expected_returns [{numpy.matrix}] -- Matriz de rendimientos anualizados esperados Nx1.
    covmat [{numpy.matrix}] -- Matriz de covarianza esperados NxN.

    Returns:
    ------
    df_ef [{pd.DataFrame}] -- DataFrame con puntos de la frontera eficiente obtenidos.
    
    """
    
    if expected_returns.shape[0] != 2:
        raise ValueError('plot_ef2 solo puede trabajar con un máximo de dos activos')
    
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, expected_returns) for w in weights]
    vols = [portfolio_volatility(w, covmat) for w in weights]
    df_ef = pd.DataFrame({'Returns':rets, 'Volatility':vols})
        
    fig = go.Figure(data=go.Scatter(x=df_ef['Volatility'], y=df_ef['Returns'], mode='lines+markers'))
    fig.update_layout(title=title)
    
    fig.show()
    
    return df_ef
