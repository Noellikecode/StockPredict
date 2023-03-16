import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import statsmodels.api as sm
import numpy as np
import yahoo_fin.stock_info as si
import datetime
# Define the stock symbol and time period
symbol = str(
    input("Please enter a stock symbol to get a price prediction(e.g AAPL):"))
# starting date
stock_data = si.get_data(symbol)
start_date = stock_data.index.min()
formatted_date = start_date.strftime("%Y-%m-%d")
start_date = formatted_date
# ending date
current_date = datetime.datetime.now()
next_date = current_date + datetime.timedelta(days=1)
year = next_date.year
month = next_date.month
day = next_date.day
end_date = f"{year}-{month:02}-{day:02}"


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split the data into training and testing sets."""
    n = len(X)
    if random_state is not None:
        np.random.seed(random_state)
    idx = np.random.permutation(n)
    X_train = X.iloc[idx[:int((1-test_size)*n)]]
    X_test = X.iloc[idx[int((1-test_size)*n):]]
    y_train = y.iloc[idx[:int((1-test_size)*n)]]
    y_test = y.iloc[idx[int((1-test_size)*n):]]
    return X_train, X_test, y_train, y_test


# Load the historical data into a pandas dataframe
df = yf.download(symbol, start=start_date, end=end_date)

# Get the stock price on the end date
end_price = si.get_live_price(symbol)

# Calculate technical indicators
sma20 = SMAIndicator(df['Close'], window=20).sma_indicator()
sma50 = SMAIndicator(df['Close'], window=50).sma_indicator()
rsi = RSIIndicator(df['Close'], window=14).rsi()

# Create a new dataframe with the technical indicators and the target variable
df_new = pd.DataFrame({'sma20': sma20, 'sma50': sma50, 'rsi': rsi})
df_new['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Remove missing values
df_new.dropna(inplace=True)

# Split the data into training and testing sets
X = df_new.drop('target', axis=1)
y = df_new['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a logistic regression model using statsmodels
X_train = sm.add_constant(X_train)
model = sm.Logit(y_train, X_train)
result = model.fit()

# Evaluate the model on the testing set
X_test = sm.add_constant(X_test)
y_pred = result.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)
accuracy = (y_pred_class == y_test).mean()
print(f'Accuracy: {accuracy:.2f}')

# Make a prediction for the next day's price movement
sma20_today = sma20[-1]
sma50_today = sma50[-1]
rsi_today = rsi[-1]
X_today = [[1, sma20_today, sma50_today, rsi_today]]
prediction = result.predict(X_today)[0]
if prediction > 0.5:
    print('The stock is predicted to go up tomorrow.')
else:
    print('The stock is predicted to go down tomorrow.')

# Print the stock price on the end date
end_price = df['Close'][-1]
date_today = datetime.datetime.now().strftime('%Y-%m-%d')

print(f'The stock price on {date_today} is ${end_price:.2f}.')
