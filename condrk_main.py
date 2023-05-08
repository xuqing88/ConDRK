import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pandas as pd
import random
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import confusion_matrix,classification_report
from crd.criterion import ContrastLoss_self
import argparse

def _init_fn(worker_id):
    np.random.seed(int(1))


def set_random_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_random_seed(4)

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def load_signals(datapath, subset, signals):
    signals_data = []

    for signal in signals:
        filename = f'{datapath}/{subset}/Inertial Signals/{signal}_{subset}.txt'
        signals_data.append(
            pd.read_csv(filename, delim_whitespace=True, header=None).values
        )

    # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
    return np.transpose(signals_data, (1, 2, 0))

def load_labels(label_path, delimiter=","):
    with open(label_path, 'rb') as file:
        y_ = np.loadtxt(label_path, delimiter=delimiter)
    return y_

def load_subjects(subject_path, delimiter=","):
    return np.loadtxt(subject_path, delimiter=delimiter)

def split_uci_data(subjectlist, _data, _labels, _subjects):
    data = []
    labels = []
    for i, subject_id in enumerate(subjectlist):
        print(f'Adding Subject {i + 1} -> {subject_id} of {len(subjectlist)} subjects')
        for j, subject in enumerate(_subjects):
            if subject == subject_id:
                data.append(_data[j])
                labels.append(_labels[j])

    return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}


def read_ucihar(datapath):
    signals = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
        "total_acc_x",
        "total_acc_y",
        "total_acc_z",
    ]

    label_map = [
        (1, 'Walking'),
        (2, 'Walking_Upstairs'),
        (3, 'Walking_Downstairs'),
        (4, 'Sitting'),
        (5, 'Standing'),
        (6, 'Laying')
    ]

    train_list = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    val_list= [23, 28,30]
    # val_list = random.sample(train_list, 3)
    for x in val_list:
        train_list.remove(x)

    subjects = {
        # Original train set = 70% of all subjects
        'train': train_list,
        # 1/3 of test set = 10% of all subjects
        'validation': val_list,
        # 2/3 of original test set = 20% of all subjects
        'test': [
            4, 12, 20, 2, 9, 10, 13, 18, 24
        ]
    }

    # labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
    idToLabel = [x[1] for x in label_map]

    print('Loading train')
    x_train = load_signals(datapath, 'train', signals)
    y_train = load_labels(f'{datapath}/train/y_train.txt')
    print('Loading test')
    x_test = load_signals(datapath, 'test', signals)
    y_test = load_labels(f'{datapath}/test/y_test.txt')
    print("Loading subjects")
    # Pandas dataframes
    subjects_train = load_subjects(f'{datapath}/train/subject_train.txt')
    subjects_test = load_subjects(f'{datapath}/test/subject_test.txt')

    _data = np.concatenate((x_train, x_test), 0)
    _labels = np.concatenate((y_train, y_test), 0)
    _subjects = np.concatenate((subjects_train, subjects_test), 0)
    print("Data: ", _data.shape, "Targets: ", _labels.shape, "Subjects: ", _subjects.shape)
    data = {dataset: split_uci_data(subjects[dataset], _data, _labels, _subjects)
            for dataset in ('train', 'validation', 'test')}

    return data, subjects


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss


def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

    return loss

def main(opt):
    outputfolder = './output/'
    teacher_type = 'resnet1d_wang'
    dataset_name = 'TIM' #[UCI, TIM]
    model_type = 'CNN'
    mpath = outputfolder + 'models/' + model_type + '/'
    iterations = 3
    batch_size = 64
    lrate = 1e-3
    nb_epoch = 100
    n_classes = 6

    if dataset_name == 'UCI':
        data, subjects = read_ucihar('dataset/UCI HAR Dataset')
        X_train, y_train = data['train']['inputs'], data['train']['targets'] - 1
        X_val, y_val = data['validation']['inputs'], data['validation']['targets'] - 1
        X_test, y_test = data['test']['inputs'], data['test']['targets'] - 1

        print('personID_list_train = ', subjects['train'])
        print('personID_list_validation = ', subjects['validation'])
        print('personID_list_test = ', subjects['test'])
        input_shape = X_train[0].shape
        X_train = torch.from_numpy(X_train).float().permute(0, 2, 1)
        y_train = torch.from_numpy(y_train).long()
        X_val = torch.from_numpy(X_val).float().permute(0, 2, 1)
        y_val = torch.from_numpy(y_val).long()
        X_test = torch.from_numpy(X_test).float().permute(0, 2, 1)
        y_test = torch.from_numpy(y_test).long()
    elif dataset_name == 'TIM':
        all_data = torch.load('dataset/TIM Dataset/har_train_val_test_dc_dqx.pt')
        X_train = torch.from_numpy(all_data['train_data']).float().permute(0, 2, 1)
        X_val = torch.from_numpy(all_data['valid_data']).float().permute(0, 2, 1)
        X_test = torch.from_numpy(all_data['test_data']).float().permute(0, 2, 1)
        y_train = torch.from_numpy(all_data['train_labels']).long()
        y_val = torch.from_numpy(all_data['valid_labels']).long()
        y_test = torch.from_numpy(all_data['test_labels']).long()
        input_shape = X_train[0].shape
        nb_epoch =200

    train_dataset = ClassifierDataset(X_train, y_train)
    val_dataset = ClassifierDataset(X_val, y_val)
    test_dataset = ClassifierDataset(X_test, y_test)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    beta_list = [0.8]
    n_batch_list = [4]

    # Load Teacher model
    if teacher_type == 'resnet1d_wang':
        from models.resnet1d import resnet1d_wang
        if dataset_name == 'UCI':
            model_t = resnet1d_wang(num_classes=n_classes, input_channels=input_shape[1]).to(device)
            model_t_name = outputfolder + 'models/' + teacher_type + '/' + "UCI_best_teacher_model.pt"
        elif dataset_name == 'TIM':
            model_t = resnet1d_wang(num_classes=n_classes, input_channels=input_shape[0]).to(device)
            model_t_name = outputfolder + 'models/' + teacher_type + '/' + "TIM_best_teacher_model.pt"
        model_t.load_state_dict(torch.load(model_t_name))
    pred_t_total = model_t(X_train.to(device)).detach()

    for n_batches in n_batch_list:
        for beta in beta_list:
            for it in range(iterations):
                # Create Student Model
                from models.resnet1d import CNN
                if dataset_name == 'UCI':
                    model = CNN(input_channels= input_shape[1] ).to(device)
                elif dataset_name =='TIM':
                    model = CNN(input_channels=input_shape[0]).to(device)

                crd = ContrastLoss_self(X_train.shape[0])
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lrate)
                scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)
                best_acc = 0
                alpha = 0.9
                kd_type = 'reg_kd'  # ['label_smooth', 'nosiy_teacher', 'reg_kd']

                for epoch in range(1,nb_epoch+1):  # loop over the dataset multiple times
                    running_loss, soft_loss_sum, hard_loss_sum, cls_loss_intra_sum,cls_loss_inter_sum = 0.0, 0.0, 0.0 , 0.0, 0.0
                    correct = 0
                    corr_val = 0
                    total_labels = 0
                    total_val = 0
                    model.train()
                    for i, (data, label) in enumerate(train_loader):
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        pred_t_pos_list = []
                        pred_t_neg_list = []

                        # teacher forward
                        pred_t = model_t(data.to(device)).detach()

                        if kd_type == 'label_smooth':
                            sigma = 0.1
                            pred_t = (1- sigma) * pred_t + sigma *torch.ones(pred_t.shape).to(device)
                        elif kd_type == 'nosiy_teacher':
                            pred_t = pred_t + torch.randn(pred_t.shape).to(device)
                        elif kd_type == 'reg_kd':

                            # Randomly sample n_batch of x_positive and x_negative instances
                            for idx in range(batch_size):
                                pos_index = (y_train == label[idx]).nonzero(as_tuple=True)[0].tolist()
                                select_ind = random.sample(pos_index, n_batches)
                                pred_t_pos_list.append(pred_t_total[select_ind])

                                neg_index = (y_train != label[idx]).nonzero(as_tuple=True)[0].tolist()
                                select_ind = random.sample(neg_index, n_batches*4)
                                pred_t_neg_list.append(pred_t_total[select_ind])

                            pred_t_pos_list = torch.mean(torch.stack(pred_t_pos_list), dim=1).to(device)
                            pred_pos = (1-beta) * pred_t_pos_list + beta * pred_t
                            pred_neg = torch.stack(pred_t_neg_list).to(device)

                        # Student forward + backward + optimize
                        pred_s = model(data.to(device))
                        outputs = torch.nn.functional.softmax(pred_s,dim=1)
                        hard_loss = criterion(outputs, label.squeeze().to(device))

                        crd_loss = crd(pred_s,pred_pos,pred_neg)
                        loss = alpha * hard_loss + (1 - alpha) * crd_loss
                        _, prediction = torch.max(outputs, 1)
                        correct += prediction.detach().cpu().eq(label.squeeze()).sum().item()
                        total_labels += prediction.shape[0]
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        hard_loss_sum += hard_loss.item()
                        cls_loss_intra_sum += crd_loss.item()

                    scheduler.step()
                    with torch.no_grad():
                        for val_data, val_label in val_loader:
                            val_output = model(val_data.to(device))
                            val_output = torch.nn.functional.softmax(val_output, dim=1)
                            _, prediction = torch.max(val_output, 1)
                            corr_val += prediction.detach().cpu().eq(val_label.squeeze()).sum().item()
                            total_val += val_label.shape[0]
                    num_batches = len(train_loader)
                    avg_loss = running_loss / num_batches
                    train_acc = 100. * correct / total_labels
                    val_acc = 100. * corr_val/ total_val

                    print(
                        f'Iteration: {it} | Epoch: {epoch}, loss: {avg_loss:.3f}, hard_loss:{hard_loss_sum/ num_batches:.3f}, '
                        f'crd_loss:{cls_loss_intra_sum/ num_batches:.3f},'
                        f'train acc: {train_acc:.2f}, val acc: {val_acc:.2f}')

                    if val_acc > best_acc:
                        best_acc = val_acc
                        model_best = copy.deepcopy(model.state_dict())

                # print validation accuracy
                model.load_state_dict(model_best)
                # model.eval()
                corr_test, total_test = 0, 0
                pred_list = []
                with torch.no_grad():
                    for test_data, test_label in test_loader:
                        # test_data = test_data.permute(0,2,1).unsqueeze(1)
                        test_output = model(test_data.to(device))
                        test_output = torch.nn.functional.softmax(test_output, dim=1)
                        _, prediction = torch.max(test_output, 1)
                        # pred_list.append(prediction.detach().cpu().tolist())
                        pred_list += prediction.detach().cpu().tolist()
                        corr_test += prediction.detach().cpu().eq(test_label.squeeze()).sum().item()
                        total_test += test_label.shape[0]

                test_acc = corr_test / total_test

                print("Confusion Matrix")
                print(confusion_matrix(y_test,pred_list))
                print(classification_report(y_test,pred_list,digits=4))
                print(f'Test accuracy: {test_acc:.4f}')

                model_name = "{}fast_{}_crd_best_model_cnn_deploy_it_{}_batch_{}_beta_{}_acc_{:.4f}.pt".format(mpath, dataset_name,
                                                                                                     it, n_batches, beta, test_acc)
                torch.save(model.state_dict(), model_name)

    print('Finished Training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
