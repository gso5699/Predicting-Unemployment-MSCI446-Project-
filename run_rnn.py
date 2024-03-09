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

# Sample data for Unemployment Consumption 

def run_rnn(
    data_cleaned, data_unscaled, scaler,
    save_path, 
    writer, 
    enable_checkpoints
):

    # Create sequences and labels for training
    seq_length = 15
    X, y = [], []
    for i in range(len(data_cleaned) - seq_length):
        X.append(data_cleaned[i:i + seq_length])
        y.append(data_cleaned[i + seq_length])

    X, y = np.array(X), np.array(y)

    # Define the proportions for train, validation, and test sets
    train_ratio = 0.8
    val_ratio = 0.1
    #test_ratio = 0.1

    train_size = int(train_ratio * len(X))
    val_size = int(val_ratio * len(X))
    #test_size = len(X) - train_size - val_size

    # Split the data into training and test sets
    train_size = int(0.8 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

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

    # Hyperparameters
    input_size = X_train.shape[2]
    hidden_size = 512
    output_size = 1
    learning_rate = 0.001
    num_epochs = 2000
    batch_size = 64

    # Create data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = RNNModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        # Training loop

        for inputs, targets in train_loader:
            outputs = model(inputs)
            train_loss = criterion(outputs, targets)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    
        writer.add_scalar('Training loss per Epoch', train_loss.item(), epoch)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)

        writer.add_scalar('Validation loss per Epoch', val_loss.item(), epoch)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
        if enable_checkpoints:
            # Save checkpoint
            if (epoch) % 10 == 0:
                if not os.path.exists(os.path.join(save_path, 'checkpoint')):
                    os.makedirs(os.path.join(save_path, 'checkpoint'))
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'train_loss': train_loss.item(),
                        'val_loss': val_loss.item()},
                    os.path.join(save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch))
                    )
            
        # Save best model
        best_loss = float('inf')

        if not os.path.exists(os.path.join(save_path, 'best')):
            os.makedirs(os.path.join(save_path, 'best'))

        if val_loss < best_loss: 
            best_loss = val_loss
            best_epoch = epoch
            torch.save({
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': val_loss,
                'optimizer_state_dict': optimizer.state_dict()}, 
                os.path.join(save_path, 'best', f'best_model.pth'))

    # Get path to best model
    best_model_path = torch.load(os.path.join(save_path, 'best', f'best_model.pth'))


    # Evaluate on the test set
    # Load best model
    model.load_state_dict(best_model_path['model_state_dict'])

    model.eval() 
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_test = model(X_test_tensor).numpy()
        y_pred_test = scaler.inverse_transform(y_pred_test)
        y_test = scaler.inverse_transform(y_test)

    # Calculate RMSE
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Write a txt file defining the hyperparameters and results on test set
    text_file_path = os.path.join(save_path, 'best', f'model_info.txt')
    with open(text_file_path, 'w') as file:
        file.write(
        f"HYPERPARAMETERS\n"
        f"Val_loss = {best_model_path['best_val_loss']}\n"
        f"Best Epoch: {best_model_path['best_epoch']}/{num_epochs}\n"
        f"Hidden size: {hidden_size}\n"
        f"Output size: {output_size}\n"
        f"Batch size: {batch_size}\n\n"
        f"ERROR ON TEST SET\n"
        f"Root Mean Squared Error (RMSE): {rmse}\n"
        f"Mean Absolute Error (MAE): {mae:.2f}\n"
        f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
    )

    # Visualize predictions on test set using matplotlib
    plt.figure(figsize=(10, 6))
    plt.scatter(data_unscaled.index[train_size+val_size+seq_length:], y_test, label='Actual',alpha=0.4)
    plt.scatter(data_unscaled.index[train_size+val_size+seq_length:], y_pred_test, label='Predicted',alpha=0.4)
    plt.xlabel('Date')
    plt.ylabel('Unemployment (%)')
    plt.title('Unemployment Prediction using RNN (PyTorch)')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'figure.png'))

    # Show image on Tensorboard
    result_fig = cv2.cvtColor(cv2.imread(os.path.join(save_path, 'figure.png')), cv2.COLOR_BGR2RGB)
    writer.add_image('Actual vs Predicted Unemployment on Test Data', result_fig, dataformats='HWC')

    plt.show()

