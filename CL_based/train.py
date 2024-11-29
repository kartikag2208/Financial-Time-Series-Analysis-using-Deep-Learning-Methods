##### Harishankar M



# Import necessary libraries
import pandas as pd
import datetime as dt
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import datetime

def str_date (date):

    return date.strftime('%Y-%m-%d')


# Function to load data
def load_data(ticker,START,TODAY):
    data = yf.download(ticker, START, TODAY, interval='5m')
    data.reset_index(inplace=True)
    return data

def dataset_summary(dataset, test_dates):
    train_count = dataset['train'].shape[0]
    test_count = dataset['test'].shape[0]
    val_count = dataset['val'].shape[0]
    total_count = train_count + val_count + test_count
    
    start_test_date = test_dates[0]
    end_test_date = test_dates[-1]

    print("### Dataset Summary ###")
    print(f"Training Samples: {train_count}")
    print(f"Validation Samples: {val_count}")
    print(f"Testing Samples: {test_count}")
    print(f"Total Samples: {total_count}")
    print(f"Test Set Start Date: {start_test_date}")
    print(f"Test Set End Date: {end_test_date}")
    print("#######################")
    
    return train_count, val_count, test_count, total_count, start_test_date, end_test_date


def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x) 
        out = self.fc(hn[-1])  
        return out
    
def detect_cusum(data, threshold):
    cum_sum = 0
    regime = 0
    regimes = [0]  

    for i in range(1, len(data)):
        change = data[i] - data[i - 1]  
        cum_sum += change

        
        if abs(cum_sum) > threshold:
            regime += 1  
            cum_sum = 0  

        regimes.append(regime)

    return regimes

def data_regimes(stock, start_time, stop_time):
    START = start_time
    TODAY = stop_time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    ticker_label = stock
    df = load_data(ticker_label,START,TODAY)

    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df['Close'].values.reshape(-1, 1)

    
    df['Return'] = df['Close'].pct_change()

    
    window_size = 100  
    df['Volatility'] = df['Return'].rolling(window=window_size).std()



def calculate_r2(actuals, predictions):
    """Calculate the R² score between actual and predicted values."""
    actuals = actuals
    predictions = predictions
    return r2_score(actuals, predictions)

def plot_results(actual, predicted,test_dates):
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates,actual, label="Actual Prices")
    plt.plot(test_dates,predicted, label="Predicted Prices")
    plt.legend()
    plt.title("Actual vs Predicted Prices")
    plt.show()

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()


    
    
    
def dataset_extraction(stock, start_time, stop_time,sequence_length,train_ratio=0.8,val_ratio = 0.2):

    
    START = start_time
    TODAY = stop_time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    ticker_label = stock
    df = load_data(ticker_label,START,TODAY)

    
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    data = df['Close'].values.reshape(-1, 1)

    
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]

    
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    
    test_data_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_data_scaled, sequence_length)
    X_test, y_test = create_sequences(test_data_scaled, sequence_length)

    
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    
    val_size = int(len(X_train) * val_ratio)
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    dataset = {'train':X_train_final,'val':X_val, 'test':X_test, 'train_label':y_train_final, 'val_label':y_val, 'test_label':y_test}

    test_dates = df.index[train_size + sequence_length:]

    return dataset,scaler,test_dates

def dataset_summary(dataset, test_dates):
    train_count = dataset['train'].shape[0]
    test_count = dataset['test'].shape[0]
    val_count = dataset['val'].shape[0]
    total_count = train_count + val_count + test_count
    
    start_test_date = test_dates[0]
    end_test_date = test_dates[-1]

    print("### Dataset Summary ###")
    print(f"Training Samples: {train_count}")
    print(f"Validation Samples: {val_count}")
    print(f"Testing Samples: {test_count}")
    print(f"Total Samples: {total_count}")
    print(f"Test Set Start Date: {start_test_date}")
    print(f"Test Set End Date: {end_test_date}")
    print("#######################")
    
    return train_count, val_count, test_count, total_count, start_test_date, end_test_date


def time_dataset_concatenate(data_1, data_2):

    """We only concatenate the training dataset, the testing and validation data will be that of data 2"""

    dataset = data_2

    dataset['train'] = torch.cat((data_1['train'],data_2['train']),dim=0)
    dataset['train_label'] = torch.cat((data_1['train_label'],data_2['train_label']),dim=0)

    return dataset


def training(dataset,Model,scaler,test_dates):

    X_train_final = dataset['train'] 
    X_val = dataset['val'] 
    X_test = dataset['test'] 

    y_train_final = dataset['train_label'] 
    y_val = dataset['val_label'] 
    y_test = dataset['test_label']

    model = Model

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    train_losses = []  
    val_losses = []    

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        train_predictions = model(X_train_final) 
        train_loss = criterion(train_predictions, y_train_final)  
        train_loss.backward()  
        optimizer.step()  
        train_losses.append(train_loss.item())  
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val) 
            val_loss = criterion(val_predictions, y_val)  
            val_losses.append(val_loss.item())  
        
        # Print losses for every epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss.item():.4f}, "
            f"Validation Loss: {val_loss.item():.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    # test_dates = df.index[train_size + sequence_length:]
    
    predicted = scaler.inverse_transform(test_predictions.cpu().numpy())
    actual = scaler.inverse_transform(y_test.cpu().numpy())

    # Calculate R2 score
    actual_numpy = np.array(actual)
    predicted_numpy = np.array(predicted)
    r2 = r2_score(actual, predicted)
    profit_percent = calculate_profit(predicted_numpy,actual_numpy)

    print(f"Profit Percent: {profit_percent}")
    
    print(f"R2 Score: {r2:.4f}")

    # Plot results

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates,actual, label='Actual Prices')
    plt.plot(test_dates,predicted, label='Predicted Prices')
    plt.legend()
    plt.title("Actual vs Predicted Prices")
    plt.show()

    return model


import numpy as np

def calculate_profit(predicted_prices, actual_prices):
    
    
    predicted_prices = np.array(predicted_prices)
    actual_prices = np.array(actual_prices)
    
    
    signals = np.sign(predicted_prices[1:] - actual_prices[:-1])
    
    
    returns = signals * (actual_prices[1:] - actual_prices[:-1])
    
    
    total_profit = np.sum(returns)
    
   
    initial_investment = actual_prices[0]
    normalized_profit = (total_profit / initial_investment) * 100
    
    return normalized_profit


