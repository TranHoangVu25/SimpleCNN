import torch
import torch.nn as nn


class SimpleNeuralNetWork(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()
        self.flatten = nn.Flatten() # thực hiện làm phẳng ảnh từ 3 chiều về 1 chiều để fit vs model
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 32, out_features=256),  # đây là fully connected layer
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),  # đây là fully connected layer
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),  # đây là fully connected layer
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),  # đây là fully connected layer
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),  # đây là fully connected layer
            nn.ReLU()
        )
    def forward(self,x):
       x =  self.flatten(x)
       x = self.fc1(x)
       x = self.fc3(x)
       x = self.fc4(x)
       x = self.fc5(x)
       x = self.fc6(x)
       return x

class SimpleCNN(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()
        self.conv1 = self.make_block(input_chanel=3,output_chanel=8)
        self.conv2 = self.make_block(input_chanel=8,output_chanel=16)
        self.conv3 = self.make_block(input_chanel=16,output_chanel=32)
        self.conv4 = self.make_block(input_chanel=32,output_chanel=64)
        self.conv5 = self.make_block(input_chanel=64,output_chanel=128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4608, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
        )



    def make_block(self,input_chanel,output_chanel):
        return nn.Sequential(
        nn.Conv2d(in_channels=input_chanel,out_channels=output_chanel,kernel_size=3,stride=1,padding="same"),
        nn.BatchNorm2d(num_features=output_chanel),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=output_chanel,out_channels=output_chanel,kernel_size=3,stride=1,padding="same"),
        nn.BatchNorm2d(num_features=output_chanel),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
# làm phẳng

        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.flatten(x)
# or choose view function as:
#         x = x.view(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = SimpleCNN()
    input_data = torch.rand(8,3,200,200)
    result = model(input_data)
    print(result.shape)
# B (Batch_size), C (chanel), H (height), W (width)


