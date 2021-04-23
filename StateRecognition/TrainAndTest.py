import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import cv2
from torch.autograd import Variable
from PrepareData import store_img_in_array
from StateRecCNN import StateCNN
from GetFeature import get_edge
from PrepareData import states


def train():
    BATCH_SIZE = 32
    train_data, test_data = store_img_in_array()
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    saving_path = './model/sccnn.pth'  # Where to find the saved model
    model = StateCNN()
    learning_rate = 0.001  # Set the learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Select ADAM as the optimizer
    loss_func = nn.CrossEntropyLoss()  # Select CrossEntropyLoss as the loss function

    num_epoch = 10
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

            if (batch_idx + 1) % 100 == 0:
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
                if epoch == 0 and batch_idx == 99:  # Print the table attributes
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


def cv_show(img, s):
    cv2.imshow(s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def single_test(file_name):
    saving_path = './model/sccnn.pth'  # Where to find the saved model
    model = StateCNN()
    model.load_state_dict(torch.load(saving_path))    # Load the saved model

    print("\n=============Single image classification started=============")
    print("Input_file: " + file_name)
    path = "./test_origin/" + file_name                 # Get the image PATH
    img = cv2.imread(path, cv2.IMREAD_COLOR)                 # Get the image (.png)
    img = cv2.resize(img, (300, 150), interpolation=cv2.INTER_AREA)
    img = get_edge(img, True)

    # cv_show(img, "test image")

    image_tensor = torch.tensor(img, dtype=torch.int)     # Transform image into a tensor
    image_tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float())
    image_tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float())

    with torch.no_grad():
        model.eval()
        output = model(image_tensor)          # Get the output tensor
        _, predicted = torch.max(output, 1)         # Get the index of the predicted label
        pre_name = states[predicted[0]]            # Get the predicted label
        print("Predicted_name: " + pre_name + "\n=============================================================")


# train()
single_test('GA08.png')

