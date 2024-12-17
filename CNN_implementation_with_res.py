import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import arff
import random
import matplotlib.pyplot as plt


# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def load_arff(filename):
    """Load ARFF file into a pandas DataFrame."""
    data, meta = arff.loadarff(filename)
    df = pd.DataFrame(data)
    # Decode byte strings to UTF-8
    for col in df.select_dtypes(include=['object', 'string']):
        df[col] = df[col].str.decode('utf-8')
    return df


def preprocess_data(df, label_column):

    df = df.dropna()

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Encode categorical features
    for col in df.select_dtypes(include=['object', 'string']):
        if col != label_column:  # Exclude label column
            df[col] = le.fit_transform(df[col].astype(str))

    # Split features and target
    X = df.drop(columns=[label_column]).values
    y = df[label_column].values

    # Encode target column if it's not numeric
    if not np.issubdtype(y.dtype, np.number):
        y = le.fit_transform(y)

    return X, y


class SimplestCNN1D(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        A 1D CNN model with residual (skip) connections.
        :param input_size: The number of input features.
        :param num_classes: Number of output classes.
        """
        super(SimplestCNN1D, self).__init__()
        self.flatten_size = input_size

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)

        # Residual connection convolution (optional adjustment to match output size)
        self.residual = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=1, padding=0)

        # Final convolutional layer to output classes
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Reshape input to [batch, channels, sequence_length]
        x = x.view(-1, 1, self.flatten_size)

        # First convolution
        out = self.conv1(x)
        out = self.relu(out)

        # Second convolution with residual connection
        residual = self.residual(x)  # Adjust input to match channels
        out = self.conv2(out) + residual  # Add skip connection
        out = self.relu(out)

        # Final output layer
        out = self.conv3(out)
        out = out.mean(dim=-1)  # Global average pooling

        return out

def train(model, X_train, y_train, criterion, optimizer, epochs=800):
    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for i in range(len(X_train)):
            optimizer.zero_grad()
            inputs = X_train[i]
            label = y_train[i]
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        avg_loss = epoch_loss / len(X_train)
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if epoch % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Plot Loss and Accuracy curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_and_train(df, label_column, dataset_name, epochs=None, lr=None):
    print(f"\n=== Training on {dataset_name} ===")
    X, y = preprocess_data(df, label_column)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare PyTorch datasets
    X_train = [X_train[i].unsqueeze(0) for i in range(X_train.size(0))]
    y_train = [y_train[i].unsqueeze(0) for i in range(y_train.size(0))]

    # Define model, loss, and optimizer
    model = SimplestCNN1D(input_size=X_train[0].shape[1], num_classes=len(torch.unique(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr if lr else 0.005)

    # Train the model
    train(model, X_train, y_train, criterion, optimizer, epochs=epochs if epochs else 500)


# Combine all datasets
def combine_datasets(*dfs):
    """
    Combine multiple datasets into a single DataFrame.
    """
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


# Load datasets
adolescent_df = load_arff('Autism-Adolescent-Data.arff')
adult_df = load_arff('Autism-Adult-Data.arff')
child_df = load_arff('Autism-Child-Data.arff')

# # Train on individual datasets
# process_and_train(adolescent_df, label_column='Class/ASD', dataset_name="Adolescent Dataset", epochs=300, lr=0.006)
# process_and_train(adult_df, label_column='Class/ASD', dataset_name="Adult Dataset", epochs=500)
# process_and_train(child_df, label_column='Class/ASD', dataset_name="Child Dataset", epochs=120, lr=0.005)

# Combine all datasets and train
combined_df = combine_datasets(adolescent_df, adult_df, child_df)
process_and_train(combined_df, label_column='Class/ASD', dataset_name="Combined Dataset", epochs=200, lr=0.005)