import torch
from torch import nn

class st_cnn_gap_5(nn.Module):
    def __init__(self,input_length=1000):
        if input_length==1000:
            self.hid_dim = 974
        elif input_length == 250:
            self.hid_dim =225
        elif input_length == 100:
            self.hid_dim = 75
        super(st_cnn_gap_5, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1),stride=1)
        )

        self.convC1 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(7,1))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1),stride=1)
        )

        self.convC2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(6,1))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,1))
        self.norm3 = nn.BatchNorm2d(64)
        self.relu_pool3 = nn.Sequential(nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=(2,1),stride=1))

        self.convE1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4,1))

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4,1),stride=1)
        )

        self.convE2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1))

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1))
        self.norm5 = nn.BatchNorm2d(64)

        self.relu_pool5 = nn.Sequential(nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=(2,1),stride=1))

        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 12), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(self.hid_dim, 1))
            # nn.MaxPool2d(kernel_size=(2, 1), stride=1)
        )

        self.flatten = nn.Flatten() #(1,62336). #(1, 64)

        self.dense8 = nn.Linear(in_features=64, out_features=128)
        self.relu_drop8 = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(p=0.1)
        )
        self.dense9 = nn.Linear(in_features=128, out_features=64)
        self.relu_drop9 = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(p=0.15)
        )
        self.output = nn.Sequential(nn.Linear(in_features=64, out_features=5),
                                    # nn.Sigmoid(),
        )


    def forward(self, x):

        pool1 = self.block1(x)
        convC1 = self.convC1(pool1)

        pool2 = self.block2(pool1)  #(1,32,988,12)
        convC2 = self.convC2(convC1) #(1,64,984,12))

        conv3 = self.conv3(pool2) #(1, 64,984,12)
        norm3 = self.norm3(conv3) #(1, 64,984,12)
        conv3 = convC2 + norm3 # Skip connection, (1, 64, 984, 12)
        pool3 = self.relu_pool3(conv3) #(1, 64, 983, 12)

        convE1 = self.convE1(pool3) #(1, 32,980,12 )
        pool4 = self.block4(pool3) #(1,64 ,978,12)
        convE2 = self.convE2(convE1) #(1,64,976,12)

        conv5 = self.conv5(pool4)  # (1, 64, 976, 12)
        norm5 = self.norm3(conv5)  # (1, 64, 976 ,12)
        conv5 = convE2 + norm5      # (1, 64, 976 ,12)
        pool5 = self.relu_pool5(conv5)   # (1, 64, 975 ,12)

        pool6 = self.block6(pool5) # (1,64,974,1)
        flat7 = self.flatten(pool6) #(1, 62336)
        dense8 = self.dense8(flat7)
        norm8 = nn.functional.normalize(dense8)
        drop8 = self.relu_drop8(norm8)
        dense9 = self.dense9(drop8)
        norm9 = nn.functional.normalize(dense9)
        drop9 = self.relu_drop9(norm9)
        output = self.output(drop9)

        return output


def main():
    # input shape (1, 12, 1000, 1)
    from thop import profile
    device = torch.device("cpu")
    input = torch.randn(1, 1, 100, 12).to(device)
    model = st_cnn_gap_5(input_length=100)
    flops, _ = profile(model, inputs=(input,))


    model = st_cnn_gap_5(input_length=100)
    # print(model)
    # input = torch.randn(1, 1, 1000, 12)
    # y_pred = model(input)
    from torchsummary import summary
    input_shape = (1, 100, 12)
    device = torch.device("cuda:0")
    summary(model.to(device), input_shape)

    print("{} | Flops ={} Billion".format('st_cnn_gap_5', flops / 1e9))


if __name__ == "__main__":
    main()
