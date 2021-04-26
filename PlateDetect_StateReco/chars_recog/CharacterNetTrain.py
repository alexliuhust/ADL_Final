import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import cv2
from chars_recog.CharacterCNN import Net
from chars_recog.CharacterPreData import store_img_in_array, classes


def train():
    BATCH_SIZE = 16
    epochs = 20
    learning_rate = 0.003
    train_data, test_data = store_img_in_array()
    trainloader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels_train = data

            optimizer.zero_grad()

            outputs_train = net(inputs)

            loss = criterion(outputs_train, labels_train)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 400 == 399:
                _, predicted = outputs_train.max(1)
                total_train += labels_train.size(0)
                correct_train += predicted.eq(labels_train).sum().item()

                test_loss = 0.0
                correct_test = 0
                total_test = 0
                with torch.no_grad():
                    for images, labels in testloader:
                        outputs_test = net(images)
                        loss_test = criterion(outputs_test, labels)
                        test_loss += loss_test.item()
                        _, predicted = outputs_test.max(1)
                        total_test += labels.size(0)
                        correct_test += predicted.eq(labels).sum().item()

                print('[%d ,%5d]loss:%.3f\tTrain Acc:%.3f\t\tTest Acc:%.3f' % (epoch + 1, i + 1,
                                                                               running_loss / 200,
                                                                               float(100. * correct_train) / float(
                                                                                   total_train),
                                                                               float(100. * correct_test) / float(
                                                                                   len(testloader.dataset))))
                running_loss = 0.0
    print('finished training!')
    PATH = './model/net.pth'
    torch.save(net.state_dict(), PATH)
    print('Model Saved')


def load_test(img):
    net = Net()
    net.load_state_dict(torch.load('./model/net.pth'))

    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    image_tensor = torch.tensor(img, dtype=torch.int)  # Transform image into a tensor
    image_tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float())
    image_tensor = Variable(torch.unsqueeze(image_tensor, dim=0).float())

    with torch.no_grad():
        net.eval()
        oputs = net(image_tensor)
        _, predicted = torch.max(oputs.data, 1)
        # print('Predicted: ', " ".join('%5s' % classes[predicted[0]]))
    return classes[predicted]


# train()  # Train the network and save the parameters

