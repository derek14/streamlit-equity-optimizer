import streamlit as st
from utils import download_ticker_data, ordered_dict_to_dataframe, plot_covariance, show_efficient_froniter, convert_df_to_csv
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from datetime import datetime

valid_ticker_list = ['SPY', 'EWJ', 'EWC', 'QQQ', 'INDA', 'EWU', 'BND', 'VNM', 'XLE']
st.title("Portfolio Optimizer (For Alon Internal Use ONLY)")

st.header("Enter Your Inputs")
st.caption("We are using data provided by Yahoo Finance.")
selected_pairs = st.multiselect(
    'What assetss do you want to include?',
    valid_ticker_list,
    ['SPY', 'EWJ', 'EWC', 'QQQ', 'INDA', 'EWU', 'BND'])

period = st.text_input('Period of analysis', "1y")

ticker_df = download_ticker_data(selected_pairs, period=period)
csv = convert_df_to_csv(ticker_df)

gamma = st.number_input('L2 Regularization Gamma', min_value=0.1, max_value=2.0, value=0.2, step=0.1, format="%.1f")

st.download_button(
    label='Download CSV',
    data=csv,
    file_name='ticker_data.csv',
    mime='text/csv'
)

st.header("Risk Modelling")
mu = expected_returns.ema_historical_return(ticker_df, frequency=365)
Sigma = risk_models.exp_cov(ticker_df, frequency=365)

st.subheader("Correlation Plot")
st.caption("A correlation plot is a graphical representation of the correlation between two or more variables. It is a scatter plot with the values of one variable on the x-axis and the values of the other variable on the y-axis. Each point on the plot represents a pair of values for the two variables. The position of the point on the plot indicates the values of the two variables, and the color or size of the point indicates the strength of the correlation between the variables.")
st.caption("The correlation between two variables is a measure of the strength and direction of the relationship between them. A positive correlation means that as one variable increases, the other variable also tends to increase. A negative correlation means that as one variable increases, the other variable tends to decrease. A correlation of zero means that there is no relationship between the variables.")
fig = plot_covariance(risk_models.exp_cov(ticker_df, frequency=365), plot_correlation=True)
st.pyplot(fig)

st.subheader("Covariance Plot")
st.caption("Covariance is a statistical measure that describes the relationship between two random variables. It measures how much two variables vary together. If two variables have a positive covariance, it means that they tend to move together in the same direction. On the other hand, if two variables have a negative covariance, it means that they tend to move in opposite directions. The covariance value can range from -∞ to +∞, with a negative value indicating a negative relationship and a positive value indicating a positive relationship. The greater this number, the more reliant the relationship. Covariance is different from correlation, which measures the strength of the relationship between two variables. Covariance is an important tool in modern portfolio theory for determining what securities to put in a portfolio. A covariance matrix is a square matrix that illustrates the variance of dataset elements and the covariance between two datasets.")
fig = plot_covariance(risk_models.exp_cov(ticker_df, frequency=365), plot_correlation=False)
st.pyplot(fig)

st.header("Get the Efficient Frontier")
st.subheader("Efficient Frontier (Long Only)")
st.caption("The efficient frontier is a set of ideal or optimal portfolios that offer the highest expected return for a specific level of risk. The standard deviation of returns in a portfolio measures investment risk and consistency in investment earnings. Lower covariance between portfolio securities results in lower portfolio standard deviation. Successful optimization of the return versus risk paradigm should place a portfolio along the efficient frontier line. Portfolios that lie below the efficient frontier are sub-optimal because they do not provide enough return for the level of risk.")
st.caption("In order to diversify risks, it is important to choose uncorrelated assets. Keeping uncorrelated assets ensures that the entire portfolio is not killed by just one stray bullet. However, a portfolio consisting only of risky assets always has some risk, since most assets have some correlation because they are all subject to systemic risk.")
st.caption("If a portfolio has very few assets that are highly correlated, it is not going to be great for several reasons. First, it will not be diversified enough, which means that it will not provide enough return for the level of risk. Second, the portfolio will have a higher standard deviation of returns, which means that it will be more volatile. Third, the portfolio will not be efficient, which means that it will not offer the highest expected return for a specific level of risk.")
weights, fig = show_efficient_froniter(mu, Sigma, long_only=True, gamma=gamma)
st.pyplot(fig)
st.table(ordered_dict_to_dataframe(weights))

st.subheader("Efficient Frontier (Long/ Short)")
st.caption("""When it comes to efficient frontier analysis in a long-only sense, having few assets that are highly correlated doesn't make sense because it doesn't provide enough diversification to reduce risk. However, in a long-short manner, having few assets that are highly correlated can work because it allows for a market-neutral strategy that can profit from the difference in performance between the two assets.""")
weights, fig = show_efficient_froniter(mu, Sigma, long_only=False, gamma=gamma)
st.pyplot(fig)
st.table(ordered_dict_to_dataframe(weights))

st.header("Maximum Sharpe Ratio Portfolio")
st.caption("The maximum sharpe ratio portfolio is a portfolio that has been optimized to provide the highest Sharpe Ratio, which is a metric that compares the amount of return versus the amount of risk, based on historical data. The Sharpe Ratio is calculated by dividing the excess return of a portfolio by its volatility, which is a measure of risk. The Maximum Sharpe Ratio Portfolio is well-suited for risk-averse investors with moderate growth expectations. The Sharpe Ratio is widely used among investors to evaluate investment performance, and a higher Sharpe Ratio indicates a better risk-adjusted return. Generally, a Sharpe Ratio between 1 and 2 is considered good, a ratio between 2 and 3 is very good, and any result higher than 3 is excellent")
ef = EfficientFrontier(mu, Sigma, weight_bounds=(0, 1))
sharpe_weights = ef.max_sharpe()
ret, vol, sharpe = ef.portfolio_performance()
st.write(f"Expected annual return: {round(ret*100, 0)}%")
st.write(f"Annual volatility: {round(vol*100, 0)}%")
st.write(f"Sharpe Ratio: {round(sharpe, 2)}")
st.table(ordered_dict_to_dataframe(sharpe_weights))


st.header("Minimum Variance Portfolio")
st.caption("A minimum variance portfolio is a collection of securities that are combined to minimize the price volatility of the overall portfolio. It is an investing strategy that uses diversification to minimize risk and maximize profits. A minimum variance portfolio holds individual, volatile securities that aren't correlated with one another. For each level of return, the portfolio with the minimum risk will be selected by a risk-averse investor, creating a minimum-variance frontier - a collection of all the minimum-variance (minimum-standard deviation) portfolios. The portion of the minimum-variance curve that lies above and to the right of the global minimum variance portfolio is known as the Markowitz efficient frontier.")
ef = EfficientFrontier(mu, Sigma, weight_bounds=(0, 1))
min_vol_weights = ef.min_volatility()
ret, vol, sharpe = ef.portfolio_performance()
st.write(f"Expected annual return: {round(ret*100, 0)}%")
st.write(f"Annual volatility: {round(vol*100, 0)}%")
st.write(f"Sharpe Ratio: {round(sharpe, 2)}")
st.table(ordered_dict_to_dataframe(min_vol_weights))
