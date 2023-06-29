import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, plotting
from pypfopt import objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
import yfinance as yf
import numpy as np
import streamlit as st

@st.cache_data
def convert_df_to_csv(df):
    csv = df.to_csv(index=True).encode('utf-8')
    return csv

def download_ticker_data(selected_tickers, period="1y"):
  temp_dict = {}
  for ticker in selected_tickers:
    temp_dict[ticker] = download_single_ticker_data(ticker, period=period)["Close"]
  ticker_df = pd.DataFrame.from_dict(temp_dict)
  ticker_df.dropna(axis='columns', inplace=True)
  return ticker_df

@st.cache_data
def download_single_ticker_data(ticker, period="1y"):
   tick = yf.Ticker(ticker)
   return tick.history(period=period)
   
def ordered_dict_to_dataframe(ordered_dict):
    df = pd.DataFrame(list(ordered_dict.items()), columns=['TICKER', 'WEIGHT'])
    df = df.set_index("TICKER")
    return df

def plot_covariance(cov_matrix, plot_correlation=False, show_tickers=True, **kwargs):
    """
    Generate a basic plot of the covariance (or correlation) matrix, given a
    covariance matrix.

    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame or np.ndarray
    :param plot_correlation: whether to plot the correlation matrix instead, defaults to False.
    :type plot_correlation: bool, optional
    :param show_tickers: whether to use tickers as labels (not recommended for large portfolios),
                        defaults to True
    :type show_tickers: bool, optional

    :return: matplotlib axis
    :rtype: matplotlib.axes object
    """
    if plot_correlation:
        matrix = risk_models.cov_to_corr(cov_matrix)
    else:
        matrix = cov_matrix
    fig, ax = plt.subplots()

    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        ax.set_xticks(np.arange(0, matrix.shape[0], 1))
        ax.set_xticklabels(matrix.index)
        ax.set_yticks(np.arange(0, matrix.shape[0], 1))
        ax.set_yticklabels(matrix.index)
        plt.xticks(rotation=90)

    plotting._plot_io(**kwargs)

    return fig

def show_efficient_froniter(mu, Sigma, long_only=True, gamma=0.1, **kwargs):
  ef = EfficientFrontier(mu, Sigma, weight_bounds=(0, 1) if long_only else (None, None))
  ef_for_quadratic = ef.deepcopy()
  ef_for_plotting = ef.deepcopy()
  ef_max_sharpe = ef.deepcopy()

  ef_for_quadratic.add_objective(objective_functions.L2_reg, gamma=gamma)
  ef_for_quadratic.max_quadratic_utility()
  weights = ef_for_quadratic.clean_weights()
  fig, ax = plt.subplots()
  
  plotting.plot_efficient_frontier(ef_for_plotting, ax=ax, show_assets=False)

  # Find the tangency portfolio
  ef_max_sharpe.max_sharpe()
  ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
  ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

  # Generate random portfolios
  n_samples = 10000
  w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
  rets = w.dot(ef.expected_returns)
  stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
  sharpes = rets / stds
  ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

  # Output
  ax.set_title("Efficient Frontier with random portfolios")
  ax.legend()
  plotting._plot_io(**kwargs)
  return weights, fig