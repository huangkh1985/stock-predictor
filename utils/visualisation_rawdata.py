# src/ml/utils/visualisation.py

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------
# Visualisation utilities plotting and data representation
# -------------------------------------------------------------------------------------------

def plot_data(df, stock, ax=None):
    if not df.empty:
        if ax:
            ax.clear()
            ax.plot(df.index, df['Close'], label=stock)
            ax.set_title(f'{stock} Price Data')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
        else:
            plt.figure(figsize=(10, 7))
            plt.plot(df.index, df['Close'], label=stock)
            plt.title(f'{stock} Price Data')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.show()

def plot_data_split(train_dates, train_y, val_dates, val_y, test_dates, test_y, stock):
    # Plot the data
    plt.figure(figsize=(10, 7))

    plt.plot(train_dates, train_y, label='Train Data', color='blue')
    plt.plot(val_dates, val_y, label='Validation Data', color='green')
    plt.plot(test_dates, test_y, label='Test Data', color='red')

    plt.title(f'{stock} Stock Price - Train, Validation, Test Sets')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
