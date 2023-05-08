import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_dim,hidden_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_1 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, dilation=4),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, stride=2, padding=1, dilation=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=5)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=7, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(kernel_size=5)
        )
        self.flat = nn.Flatten()

    def forward(self, src):
        fea_1 = self.encoder_1(src)
        fea_2 = self.encoder_2(src)
        fea_3 = self.encoder_3(src)
        # features = self.flat(torch.cat((fea_1,fea_2,fea_3),dim=2))
        features = torch.cat((fea_1, fea_2, fea_3), dim=2) # [?, 12, 34]  #[?,1,12,181]

        return features


class CNN_RUL_student_stack(nn.Module):
    def __init__(self, input_dim, hidden_dim,n_class, dropout):
        super(CNN_RUL_student_stack, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        self.n_class = n_class
        self.generator = Generator(input_dim=input_dim,hidden_dim=hidden_dim)
        self.regressor= nn.Sequential(
            nn.Linear(9088, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.n_class)
        )

        self.flat = nn.Flatten()
        self.head_csra = nn.Conv2d(self.hidden_dim, self.n_class, 1, bias=False)
        self.lam = 1.0

    def forward(self, src):
        features = self.generator(src)

        # features = self.flat(features)
        # predictions = self.regressor(features)
        # return predictions.squeeze()

        x = torch.unsqueeze(features, dim=-1)
        score = self.head_csra(x) / torch.norm(self.head_csra.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)  # (1,5)
        att_logit = torch.max(score, dim=2)[0]
        # return base_logit
        return base_logit + self.lam * att_logit


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # input shape (1, 12, 1000, 1)
    from thop import profile
    device = torch.device("cpu")
    input = torch.randn(1, 12, 100).to(device)
    model = CNN_RUL_student_stack(input_dim=12,hidden_dim=64, n_class=5, dropout=0.25)
    flops, _ = profile(model, inputs=(input,))


    model = CNN_RUL_student_stack(input_dim=12,hidden_dim=64, n_class=5,dropout=0.25)
    # print(model)
    # input = torch.randn(1, 1, 1000, 12)
    # y_pred = model(input)
    from torchsummary import summary
    input_shape = (12, 100)
    device = torch.device("cuda:0")
    summary(model.to(device), input_shape)

    print("{} | Flops ={} Billion".format('dilated_cnn', flops / 1e9))
    print('parameters_count:',count_parameters(model))

if __name__ == "__main__":
    main()


