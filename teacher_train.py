import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import confusion_matrix,classification_report

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


def read_raw(type_data):

    all_filenames = []
    all_filenames += ["body_acc_x_{}.txt".format(type_data), "body_acc_y_{}.txt".format(type_data), "body_acc_z_{}.txt".format(type_data)]
    all_filenames += ["total_acc_x_{}.txt".format(type_data), "total_acc_y_{}.txt".format(type_data), "total_acc_z_{}.txt".format(type_data)]
    all_filenames += ["body_gyro_x_{}.txt".format(type_data), "body_gyro_y_{}.txt".format(type_data), "body_gyro_z_{}.txt".format(type_data)]
    train_label = "../UCI HAR Dataset/{}/y_{}.txt".format(type_data, type_data)
    subject_file = "../UCI HAR Dataset/{}/subject_{}.txt".format(type_data, type_data)
    all_data_list = []
    for each_file in all_filenames:
        full_path = "../UCI HAR Dataset/{}/Inertial Signals/{}".format(type_data, each_file)
        train_data_np = pd.read_csv(full_path, header=None, delim_whitespace=True)
        all_data_list.append(train_data_np)

    all_data_np = np.dstack(all_data_list)
    all_labels_df = pd.read_csv(train_label, header=None, delim_whitespace=True)
    all_subjects_df = pd.read_csv(subject_file, header=None, delim_whitespace=True)

    return all_data_np, all_labels_df.values, all_subjects_df.values

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


def convert_data(train_data_np, val_data_np, test_data_np):

    data = np.concatenate((train_data_np, val_data_np, test_data_np), axis=0)
    min = -128
    max = 127

    min_val = data.min(axis=0)
    max_val = data.max(axis=0)

    X_std = (data - min_val) / (max_val - min_val)
    norm_np = X_std * (max - min) + min

    norm_np = norm_np.astype(int)  # change to int
    X_train = norm_np[:train_data_np.shape[0]]
    X_val = norm_np[train_data_np.shape[0]:train_data_np.shape[0]+val_data_np.shape[0]]
    X_test= norm_np[train_data_np.shape[0]+val_data_np.shape[0]:]

    return X_train, X_val, X_test



def main():
    outputfolder = './output/'
    model_type = 'resnet1d_wang' #['resnet1d101', 'resnet1d_wang', 'xresnet1d34']
    mpath = outputfolder + 'models/'+ model_type + '/'
    if not os.path.exists(mpath):
        os.makedirs(mpath)

    dataset_name = 'TIM' #[UCI, TIM]

    iterations = 1
    batch_size = 64
    lrate = 1e-3
    nb_epoch = 200
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
        nb_epoch = 200

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

    for it in range(iterations):
        if model_type == 'xresnet1d34':
            from models.xresnet1d import xresnet1d34
            model = xresnet1d34(num_classes=n_classes, input_channels=input_shape[1], kernel_size=5).to(device)
        elif model_type == 'resnet1d_wang':
            from models.resnet1d import resnet1d_wang
            if dataset_name == 'UCI':
                model = resnet1d_wang(num_classes=n_classes, input_channels=input_shape[1]).to(device)
            elif dataset_name == 'TIM':
                model = resnet1d_wang(num_classes=n_classes, input_channels=input_shape[0]).to(device)
        elif model_type == 'resnet1d101':
            from models.resnet1d import resnet1d101
            model = resnet1d101(num_classes=n_classes, input_channels=input_shape[1], inplanes=128, kernel_size=5).to(device)
        elif model_type == "CNN":
            from models.resnet1d import CNN
            if dataset_name == 'UCI':
                model = CNN(input_channels=input_shape[1]).to(device)
            elif dataset_name == 'TIM':
                model = CNN(input_channels=input_shape[0]).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lrate)
        scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

        best_acc = 0
        for epoch in range(1,nb_epoch+1):  # loop over the dataset multiple times
            running_loss = 0.0
            correct = 0
            corr_val = 0
            total_labels = 0
            total_val = 0
            model.train()

            for i, (data, label) in enumerate(train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()
                if model_type =='AI85NET5':
                    data = data.unsqueeze(1)
                # forward + backward + optimize
                if model_type == 'resnet1d_wang':
                    outputs= model(data.to(device))
                else:
                    outputs, _ = model(data.to(device))
                outputs = torch.nn.functional.softmax(outputs,dim=1)
                loss = criterion(outputs, label.squeeze().to(device))
                _, prediction = torch.max(outputs, 1)
                correct += prediction.detach().cpu().eq(label.squeeze()).sum().item()
                total_labels += prediction.shape[0]
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()

            # model.eval()
            with torch.no_grad():
                for val_data, val_label in val_loader:
                    if model_type == 'resnet1d_wang':
                        val_output= model(val_data.to(device))
                    else:
                        val_output, _ = model(val_data.to(device))
                    val_output = torch.nn.functional.softmax(val_output, dim=1)
                    _, prediction = torch.max(val_output, 1)
                    corr_val += prediction.detach().cpu().eq(val_label.squeeze()).sum().item()
                    total_val += val_label.shape[0]
            num_batches = len(train_loader)
            avg_loss = running_loss / num_batches
            train_acc = 100. * correct / total_labels
            val_acc = 100. * corr_val/ total_val

            print(f'Iteration: {it} | Epoch: {epoch}, Loss: {avg_loss:.3f}, train acc: {train_acc:.2f}, val acc: {val_acc:.2f}')

            if val_acc > best_acc:
                best_acc = val_acc
                model_best = copy.deepcopy(model.state_dict())

        # print validation accuracy
        # Evaluate on test data set
        model.load_state_dict(model_best)
        corr_test, total_test = 0, 0
        pred_list = []
        with torch.no_grad():
            for test_data, test_label in test_loader:
                if model_type == 'resnet1d_wang':
                    test_output = model(test_data.to(device))
                else:
                    test_output, _ = model(test_data.to(device))

                test_output = torch.nn.functional.softmax(test_output, dim=1)
                _, prediction = torch.max(test_output, 1)
                pred_list += prediction.detach().cpu().tolist()
                corr_test += prediction.detach().cpu().eq(test_label.squeeze()).sum().item()
                total_test += test_label.shape[0]

        test_acc = corr_test / total_test

        print("Confusion Matrix")
        print(confusion_matrix(y_test,pred_list))
        print(classification_report(y_test,pred_list,digits=4))
        print(f'Test accuracy: {test_acc:.4f}')

        model_name = "{}{}_best_teacher_model.pt".format(mpath, dataset_name)
        torch.save(model.state_dict(), model_name)

    print('Finished Training')

if __name__ == "__main__":
    main()
