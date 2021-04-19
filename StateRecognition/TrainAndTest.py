import torch
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PrepareData import store_img_in_array
from StateRecCNN import StateCNN


BATCH_SIZE = 32
train_data, test_data = store_img_in_array()
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# for batch_idx, (inputs, targets) in enumerate(train_loader):
#     if batch_idx == 0:
#         print(inputs.size())
#         print(targets.size())
#         print(targets[10])
#         break

saving_path = './model/sccnn.pth'             # Where to find the saved model
model = StateCNN()
# model.load_state_dict(torch.load(saving_path))    # Load the saved model
learning_rate = 0.001                             # Set the learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)   # Select ADAM as the optimizer
loss_func = nn.CrossEntropyLoss()                 # Select CrossEntropyLoss as the loss function


def train():
    num_epoch = 3
    for epoch in range(num_epoch):
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradient

            outputs = model(inputs)  # Get the output
            loss = loss_func(outputs, targets)  # Calculate the loss
            loss.backward()  # Backward propagation
            optimizer.step()  # Update the weights

            train_loss += loss.item()  # Accumulate the total loss
            _, predicted = outputs.max(1)  # Get the prediction result
            total += targets.size(0)  # Accumulate the total number of the samples
            correct += predicted.eq(targets).sum().item()  # Accumulate the number of correct classification

            if (batch_idx + 1) % 5 == 0:
                # ============================== Test result: ===================================
                # Zero those result for each test
                t_ttl_loss = 0
                t_correct = 0
                t_total = 0
                with torch.no_grad():
                    for test_batch_idx, (t_inputs, t_targets) in enumerate(test_loader):
                        t_outputs = model(t_inputs)  # Get the output
                        t_loss = loss_func(t_outputs, t_targets)  # Calculate the loss

                        t_ttl_loss += t_loss.item()  # Accumulate the total loss
                        _, t_predicted = t_outputs.max(1)  # Get the prediction result
                        t_total += t_targets.size(0)  # Accumulate the total number of the samples
                        t_correct += \
                            t_predicted.eq(t_targets).sum().item()  # Accumulate the number of correct classification

                # ============================== Print result: ================================
                if epoch == 0 and batch_idx == 4:  # Print the table attributes
                    print("\nModel Training Started...")
                    print("Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest Acc")

                print(
                    '[{}/{}]\t{:.4f}\t\t{:.3f}%\t\t{:.4f}\t\t{:.3f}%'
                    .format(
                        (epoch + 1), num_epoch,  # Epoch
                        train_loss / (len(train_loader)),  # Train Loss
                        float(100. * correct) / float(total),  # Train Acc
                        t_ttl_loss / len(test_loader),  # Test Loss
                        float(100. * t_correct) / float(t_total)  # Test Acc
                    )
                )

    # Save the model
    torch.save(model.state_dict(), saving_path)
    print("Model saved in file: " + saving_path)


train()


