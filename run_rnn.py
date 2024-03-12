import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

### The code was adapted from: https://encord.com/blog/time-series-predictions-with-recurrent-neural-networks/#:~:text=Recurrent%20Neural%20Networks%20(RNNs)%20bring,genuine%20potential%20of%20predictive%20analytics.

# Create a custom dataset class for PyTorch DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Define Functions
def get_preprocessed_data():
    df = pd.read_csv('data/unemployment_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])  
    df.set_index('Date', inplace=True)
    df = df[['Unemployment']]
    df.dropna(how='any', inplace=True)

    # Preprocessing
    scaler = MinMaxScaler()
    data_cleaned = scaler.fit_transform(df)
    return(data_cleaned, df, scaler) # df returns unscaled data

def split_data(data, seq_length):
    # Create sequences and labels for training
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    X, y = np.array(X), np.array(y)

    # Define the proportions for train, validation, and test sets
    train_ratio = 0.76
    val_ratio = 0.12
    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))

    # Split the data into training and test sets
    train_size = int(0.8 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

    return(X_train, X_val, X_test, y_train, y_val, y_test)

def evaluate_model_on_test(path_to_model, save_path, writer, seq_length):
    
    data_cleaned, data_unscaled, scaler = get_preprocessed_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data=data_cleaned, seq_length=seq_length)

    # Evaluate on the test set
    model_path = torch.load(path_to_model)

    # Load best model
    model = RNNModel(input_size=1, hidden_size=model_path['hidden_size'], output_size=1)
    model.load_state_dict(model_path['model_state_dict'])

    # Evaluate model on the test data
    model.eval() 
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_test = model(X_test_tensor).numpy()
        y_pred_test = scaler.inverse_transform(y_pred_test)
        y_test = scaler.inverse_transform(y_test)

    # Print hyperparameters of the selected model to console
    print("\n==================================")
    print("HYPERPARAMETERS FOR SELECTED MODEL")
    print("Epoch:", f"{model_path['epoch']}/{model_path['total_epochs']}")
    print("Hidden Size:", model_path['hidden_size'])
    print("Learning Rate:", model_path['learning_rate'])
    print("Batch Size:", model_path['batch_size'])
    print("Sequence Length:", model_path['seq_length'])
    print("\n==================================")
    print("Training Loss:", model_path['train_loss'])
    print("Validation Loss:", model_path['val_loss'])
    # Calculate RMSE and print to console
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Visualize predictions on test set using matplotlib
    plt.figure(figsize=(10, 6))
    plt.scatter(data_unscaled.index[X_train.shape[0]+X_val.shape[0]+seq_length:], y_test, label='Actual',alpha=0.4)
    plt.scatter(data_unscaled.index[X_train.shape[0]+X_val.shape[0]+seq_length:], y_pred_test, label='Predicted',alpha=0.4)
    plt.xlabel('Date')
    plt.ylabel('Unemployment (%)')
    plt.title('Unemployment Prediction using RNN (PyTorch)')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'figure.png'))

    # Show image on Tensorboard
    result_fig = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'figure.png')), cv2.COLOR_BGR2RGB)
    writer.add_image('Actual vs Predicted Unemployment on Test Data', result_fig, dataformats='HWC')

    # Write a txt file to save the hyperparameters and error calculations
    text_file_path = os.path.join(save_path, 'best', f'model_info.txt')
    with open(text_file_path, 'w') as file:
        file.write(
        "==================================\n"
        "HYPERPARAMETERS FOR SELECTED MODEL\n"
        f"Epoch: {model_path['epoch']}/{model_path['total_epochs']}\n"
        f"Hidden Size: {model_path['hidden_size']}\n"
        f"Learning Rate: {model_path['learning_rate']}\n"
        f"Batch Size: {model_path['batch_size']}\n"
        f"Sequence Length: {model_path['seq_length']}\n"
        f"==================================\n"
        f"Training Loss: {model_path['train_loss']}\n"
        f"Validation Loss: {model_path['val_loss']}\n"
        "==================================\n"
        "ERROR ON TEST SET\n"
        f"Root Mean Squared Error (RMSE): {rmse}\n"
        f"Mean Absolute Error (MAE): {mae:.2f}\n"
        f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
        )
    plt.show()

# Main Function
def run_rnn(
    save_path, 
    writer, 
    enable_checkpoints
):
    # Hyperparameters
    input_size = 1
    hidden_size = 180
    output_size = 1
    learning_rate = 0.001
    num_epochs = 3000
    batch_size = 64
    seq_length = 28

    data_cleaned, _, _ = get_preprocessed_data()

    # Split the data
    X_train, X_val, _, y_train, y_val, _ = split_data(data=data_cleaned, seq_length=seq_length)

    # Create data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loop
    best_loss = float('inf') # Set the initial best_loss to infinity
    for epoch in range(num_epochs):

        # Training loop
        for inputs, targets in train_loader:
            outputs = model(inputs)
            train_loss = criterion(outputs, targets)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # Write to tensorboard
        writer.add_scalar('Training loss per Epoch', train_loss.item(), epoch+1)

        # Validation loop
        model.eval()
        # Make a folder called "best" to save the best model
        if not os.path.exists(os.path.join(save_path, 'best')):
            os.makedirs(os.path.join(save_path, 'best'))

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)

                # If it is the best model so far, save the model to folder
                if val_loss < best_loss: 
                    best_loss = val_loss
                    best_epoch = epoch+1
                    torch.save({
                        'epoch': best_epoch,
                        'total_epochs': num_epochs,
                        'model_state_dict': model.state_dict(),
                        'train_loss': train_loss.item(),
                        'val_loss': val_loss.item(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'input_size': input_size,
                        'hidden_size': hidden_size,
                        'output_size': output_size,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'seq_length': seq_length}, 
                        os.path.join(save_path, 'best', f'best_model.pth'))
                
                # If enable_checkpoints is enabled, save a model for every 100 epochs in "checkpoint" folder   
                if enable_checkpoints:
                    # Save checkpoint
                    if (epoch) % 100 == 0:
                        if not os.path.exists(os.path.join(save_path, 'checkpoint')):
                            os.makedirs(os.path.join(save_path, 'checkpoint'))
                        torch.save({
                                'epoch': epoch,
                                'total_epochs': num_epochs,
                                'model_state_dict': model.state_dict(),
                                'train_loss': train_loss.item(),
                                'val_loss': val_loss.item(),
                                'input_size': input_size,
                                'hidden_size': hidden_size,
                                'output_size': output_size,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size,
                                'seq_length': seq_length},
                            os.path.join(save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch))
                            )
        # Write to Tensorboard
        writer.add_scalar('Validation loss per Epoch', val_loss.item(), epoch)

        # Display the results of every 10 epochs in the console
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Get path to best model
    best_model_path = os.path.join(save_path, 'best', f'best_model.pth')

    # Evaluate the best_model on the test set
    evaluate_model_on_test(best_model_path, save_path, writer, seq_length)

