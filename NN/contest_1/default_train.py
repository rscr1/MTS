import os
import gc
import yaml
import random
import time

import cv2
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm, trange
from clearml import Task

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam, AdamW, SGD, lr_scheduler

from torchmetrics import F1Score, Recall, Precision, Accuracy, MetricCollection

import  albumentations as A 
from albumentations.pytorch import ToTensorV2


class BasicBlock(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            stride=1,
            expansion=1, 
            downsample=None):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False
            )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels * self.expansion)
        self.relu2 = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample:
            shortcut = self.downsample(shortcut)
        x += shortcut
        return self.relu2(x)

     
class MyResnet18(nn.Module):
    def __init__(
            self,
            block,
            img_channels,
            num_classes   
    ) -> None:
        super(MyResnet18, self).__init__()
        self.in_channels: int = 64
        self.expansion = 1

        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3, 
            bias=False
            )
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        layers: list = [2] * 4
        
        self.layer1 = self._make_block(block, 64, layers[0], 1)
        self.layer2 = self._make_block(block, 128, layers[1], 2)
        self.layer3 = self._make_block(block, 256, layers[2], 2)
        self.layer4 = self._make_block(block, 512, layers[3], 2)

        self.avgpool = nn. AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features=512 * self.expansion,
            out_features=num_classes
        )

    def _make_block(
            self, 
            block,
            out_channels,
            blocks,
            stride
            ):
        downsample = None
        layers: list = []
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                self.expansion,
                downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                self.expansion
            ))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x =self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x.flatten(1))

        return x
    

class TrainDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            data,
            transforms=None, 
            target_transforms=None
            ) -> None:

        self.dataset_path = dataset_path
        self.data = data
        self.transforms = transforms
        self.target_transforms = target_transforms


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data.iloc[idx]
        image = cv2.imread(os.path.join(self.dataset_path, image), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        if self.target_transforms:
            label = self.target_transforms(label)

        return image, label


class TestDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            data,
            transforms=None
            ) -> None:

        self.dataset_path = dataset_path
        self.data = data
        self.transforms = transforms


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        path = self.data.iloc[idx]['filename']

        image = cv2.imread(os.path.join(self.dataset_path, path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, path


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def save_training(losses, metrics, metrics_per_class, best_epoch, best_metric, losses_path, train_metrics_path, val_metrics_path, dataframe_path):
    metrics_names = ['F1Score', 'Recall', 'Precision', 'Accuracy']
    loss_at_best_epoch = losses['val'][best_epoch]
    if losses_path is not None:
        plt.figure()
        plt.plot(losses['train'], label='Train Loss')
        plt.plot(losses['val'], label='Valid Loss')
        plt.title('Train/Valid losses')
        # plt.text(x=0, y=0.1, s=f'Loss at best epoch: {loss_at_best_epoch:.4f}', fontsize=10)
        plt.grid()
        plt.legend()
        plt.savefig(losses_path)
        plt.close()
    if train_metrics_path is not None:
        plt.figure()
        for metric_name in metrics_names:
            plt.plot(metrics['train'][metric_name], label=metric_name)
        plt.title('Train metrics')
        plt.grid()
        plt.legend()
        plt.savefig(train_metrics_path)
        plt.close()
    if val_metrics_path is not None:
        plt.figure()
        for metric_name in metrics_names:
            plt.plot(metrics['val'][metric_name], label=metric_name)
        plt.title('Valid metrics')
        # plt.text(x=0, y=0.1, s=f'best epoch: {best_epoch}\nbest metric: {best_metric:.4f}', fontsize=10)
        plt.grid()
        plt.legend()
        plt.savefig(val_metrics_path)
        plt.close()
    if dataframe_path is not None:
        metrics_per_class = np.array(metrics_per_class)
        epochs = metrics_per_class.shape[1]
        data = {
            'epoch': [], 
            'metric': metrics_names * epochs,
            'mean': metrics_per_class.mean(2).flatten('F')
            }
        [data['epoch'].extend([i] * 4) for i in range(epochs)]
        for i in range(metrics_per_class.shape[2]):
            data[i] = metrics_per_class[:, :, i].flatten('F')

        pd.DataFrame(data).to_csv(dataframe_path, index=False)


def print_train_metrics(rank, losses, epoch, metrics, metrics_names, phase):
    symb = 'T' if phase == 'train' else 'V'
    symb = symb.ljust(len(str(epoch)), ' ')
    part_1 = f'[GPU:{rank} | Phase:{symb}]'
    part_2 = str(round(losses[phase][-1], 4)).ljust(6, ' ')
    part_3 = ' '.join(str(round(metrics[phase][name][-1], 4)).ljust(len(name), ' ') for name in metrics_names)
    print(part_1, part_2, part_3)


def print_stat(device, optimizer, batch, start_time):
    part_1 = f'[GPU:{device[-1]}] LR='
    part_2 = optimizer.param_groups[0]['lr']
    part_3 = ' BS='
    part_4 = len(batch[0])
    part_5 = f' Time={round(time.time() - start_time, 2)}s'
    print(part_1, part_2, part_3, part_4, part_5, sep='')
    

def train_loop(
        model, 
        dataloader, 
        loss_func, 
        optimizer, 
        metric, 
        main_metric,
        device,
        classes, 
        patience, 
        scheduler=None, 
        epochs=None, 
        best_model_path=None, 
        last_model_path=None, 
        losses_path=None, 
        train_metrics_path=None, 
        val_metrics_path=None, 
        dataframe_path=None
        ) -> None:

    metrics_names = ['F1Score', 'Recall', 'Precision', 'Accuracy']
    main_metric_index = metrics_names.index(main_metric)
    metrics = {
        'train': {metrics_names[i]: [] for i in range(4)},
        'val': {metrics_names[i]: [] for i in range(4)}
    }
    losses = {
        'train': [],
        'val': []
    }
    metrics_per_class = [[] for _ in range(4)]
    best_metric = 0
    best_epoch = 0
    
    for epoch in trange(epochs, leave=False, desc='Epochs'):
        start_time = time.time()
        for phase in ('train', 'val'):
            total_loss = torch.zeros(1, device=device)
            total_metric_per_class = torch.zeros((4, classes), device=device)

            for batch in dataloader[phase]:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                model.train(phase == 'train')
                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = loss_func(outputs, labels)

                total_loss += loss
                metrics_dict = metric(outputs.argmax(1), labels)
                for i in range(4):
                    total_metric_per_class[i] += torch.nan_to_num(metrics_dict[metrics_names[i]], 0)

            epoch_loss = total_loss / len(dataloader[phase])
            losses[phase].append(epoch_loss.item())
            
            epoch_metric_per_class = total_metric_per_class / len(dataloader[phase])
            epoch_metric_per_class = epoch_metric_per_class.detach().cpu().numpy()
            epoch_metrics = epoch_metric_per_class.mean(1)

            gc.collect()
            torch.cuda.empty_cache()

            for i in range(4):
                metrics[phase][metrics_names[i]].append(epoch_metrics[i])

            if phase =='train':
                print(f'\n[GPU:{device[-1]} | Epoch:{epoch}]', 'Loss'.ljust(6, ' '), ' '.join(metrics_names))
                print_train_metrics(device[-1], losses, epoch, metrics, metrics_names, phase)

            if phase == 'val':
                print_train_metrics(device[-1], losses, epoch, metrics, metrics_names, phase)
                print_stat(device, optimizer, batch, start_time)

                if scheduler is not None:
                    scheduler.step(epoch_metrics[main_metric_index])

                if epoch_metrics[main_metric_index] > best_metric:
                    best_metric = epoch_metrics[main_metric_index]
                    best_epoch = epoch
                    torch.save(model.module.state_dict(), best_model_path)
                    print('Best score, weights have been saved')

                if dataframe_path is not None:
                    [metrics_per_class[i].append(epoch_metric_per_class[i]) for i in range(4)]

                if epoch - best_epoch == patience:
                    save_training(
                        losses, metrics, metrics_per_class, 
                        best_epoch, best_metric, 
                        losses_path, train_metrics_path, val_metrics_path, dataframe_path
                    )
                    torch.save(model.module.state_dict(), last_model_path)
                    print('Early stopping!!! Last weights have been saved')
                    print(f'Training completed, best epoch: {best_epoch}, best {main_metric}: {best_metric:.4f}')
                    print()
                    return None
    
    save_training(
        losses, metrics, metrics_per_class, 
        best_epoch, best_metric, 
        losses_path, train_metrics_path, val_metrics_path, dataframe_path
    )
    torch.save(model.module.state_dict(), last_model_path)
    print('Last weights have been saved')
    print(f'Training completed, best epoch: {best_epoch}, best {main_metric}: {best_metric:.4f}')
    print()


def test_loop(model, dataloader, device, submission_path):
    start_time = time.time()
    model.eval()
    data = {'filename': [], 'label': [], 'probs': []}
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False, desc='Batches'):
            images, paths = batch
            images = images.to(device)
            logits = model(images)
            outputs = logits.argmax(1).detach().cpu().tolist()
            probs = torch.softmax(logits, dim=-1).max().detach().cpu().tolist()

            data['filename'].extend(paths)
            data['label'].extend(outputs)
            data['probs'].extend(probs)

    mapper = {j: i for i, j in {'Koffing': 0,'Venonat': 1,'Nidoking': 2,'Victreebel': 3, 'Persian': 4,'Ekans': 5,'Jigglypuff': 6,'Vulpix': 7,'Lickitung': 8,'Cloyster': 9,'Paras': 10,'Pidgeotto': 11,'Rapidash': 12,'Ditto': 13,'Clefairy': 14,'Charmander': 15,'Kakuna': 16,'Flareon': 17,'Charmeleon': 18,'Raichu': 19,'Seel': 20,'Omanyte': 21,'Machamp': 22,'Tauros': 23,'Primeape': 24,'Ponyta': 25,'Hitmonchan': 26,'Mankey': 27,'Electabuzz': 28,'Chansey': 29,'Hypno': 30,'Gloom': 31,'Nidoqueen': 32,'Vaporeon': 33,'Rattata': 34,'Metapod': 35,'Sandshrew': 36,'Kabuto': 37,'Eevee': 38,'Pinsir': 39,'Arbok': 40,'Kabutops': 41,'Seadra': 42,'Lapras': 43,'Arcanine': 44,'Poliwhirl': 45,'Gyarados': 46,'Jynx': 47,'Venusaur': 48,'Venomoth': 49,'Kadabra': 50,'Butterfree': 51,'Seaking': 52,'Tangela': 53,'Slowpoke': 54,'Growlithe': 55,'Marowak': 56,'Spearow': 57,'Electrode': 58,'Drowzee': 59,'Dewgong': 60,'Squirtle': 61,'Magnemite': 62,'Exeggcute': 63,'Bulbasaur': 64,'Golbat': 65,'Weezing': 66,'Muk': 67,'MrMime': 68,'Porygon': 69,'Magmar': 70,'Sandslash': 71,'Raticate': 72,'Vileplume': 73,'Wigglytuff': 74,'Starmie': 75,'Wartortle': 76,'Gastly': 77,'Voltorb': 78,'Horsea': 79,'Alakazam': 80,'Psyduck': 81,'Fearow': 82,'Parasect': 83,'Blastoise': 84,'Tentacruel': 85,'Pidgeot': 86,'Mewtwo': 87,'Jolteon': 88,'Poliwrath': 89,'Scyther': 90,'Moltres': 91,'Nidorino': 92,'Pikachu': 93,'Slowbro': 94,'Diglett': 95,'Grimer': 96,'Gengar': 97,'Dugtrio': 98,'Omastar': 99,'Staryu': 100,'Golduck': 101,'Dodrio': 102,'Machop': 103,'Aerodactyl': 104,'Nidorina': 105,'Abra': 106,'Kangaskhan': 107,'Snorlax': 108,'Rhydon': 109,'Ivysaur': 110,'Zapdos': 111,'Rhyhorn': 112,'Bellsprout': 113,'Dragonair': 114,'Shellder': 115,'Graveler': 116,'Machoke': 117,'Zubat': 118,'Onix': 119,'Alolan Sandslash': 120,'Cubone': 121,'Caterpie': 122,'Haunter': 123,'Farfetchd': 124,'Dragonite': 125,'Weepinbell': 126,'Oddish': 127,'Poliwag': 128,'Kingler': 129,'Pidgey': 130,'Beedrill': 131,'Magikarp': 132,'Krabby': 133,'Tentacool': 134,'Goldeen': 135,'Geodude': 136,'Meowth': 137,'Exeggutor': 138,'Dratini': 139,'Clefable': 140,'Mew': 141,'Weedle': 142,'Articuno': 143,'Hitmonlee': 144,'Doduo': 145,'Golem': 146,'Magneton': 147,'Secretok': 148,'Ninetales': 149,'Charizard': 150}.items()}
    
    data = pd.DataFrame(data)
    data['label'] = data['label'].map(mapper)
    data.to_csv(submission_path, sep=',', index=False)
    print(f'Testing completed, time={round(time.time() - start_time, 2)}s')
    print(f'Submission has been saved to: {submission_path}')


def train(CFG):
    set_seed(CFG['seed'])
    save_path = os.path.join(CFG['save_path'], CFG['name'])
    if os.path.exists(save_path) and CFG['train']:
        counter = 1
        while os.path.exists(save_path):
            save_path = os.path.join(CFG['save_path'], CFG['name'] + str(counter))
            counter += 1

        CFG['name'] = CFG['name'] + str(counter - 1)

    print(f'Results will be saved to {save_path}')
    os.makedirs(save_path)

    best_model_path = f'{save_path}/best.pth'
    last_model_path = f'{save_path}/last.pth'
    train_metrics_path = f'{save_path}/train_metrics.png'
    val_metrics_path = f'{save_path}/val_metrics.png'
    losses_path = f'{save_path}/losses.png'
    dataframe_path = f'{save_path}/metrics.csv'
    submission_path = f'{save_path}/submission.csv'
    config_path = f'{save_path}/config.yaml'

    with open(config_path, 'w') as file:
        yaml.dump(CFG, file, default_flow_style=False)
        file.close()

    print(f'Config file has been saved to: {config_path}')

    clearml = Task.init(
        project_name='private/AkhmetzyanovD/' + CFG['save_path'].split('/')[-2], 
        task_name=CFG['name'],
        tags=[CFG['model'], CFG['loss'], CFG['optim']],
        reuse_last_task_id=False,
    )
    clearml.set_parameters_as_dict(CFG)

    train_transforms = A.Compose([
        A.Resize(height=CFG['imgsz'], width=CFG['imgsz'], interpolation=CFG['interpolation'], p=1.0),
        A.HorizontalFlip(p=CFG['hf']),
        A.VerticalFlip(p=CFG['vf']),
        # A.ShiftScaleRotate(
        #     shift_limit=(-CFG['sh'], CFG['sh']), 
        #     scale_limit=(-CFG['sc'], CFG['sc']), 
        #     rotate_limit=(-CFG['r'], CFG['r']),
        #     border_mode=1
        #     ),
        # A.PixelDropout(p=CFG['pd']),
        # A.CoarseDropout(10, 16, 16, 0, 0, 0, p=CFG['cd']),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0)
        ])

    val_transforms = A.Compose([
        A.Resize(height=CFG['imgsz'], width=CFG['imgsz'], interpolation=CFG['interpolation'], p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0)
        ])

    # data
    train_data = pd.read_csv(CFG['train_path'], sep=',', index_col=False)
    val_data = pd.read_csv(CFG['val_path'], sep=',', index_col=False)
    test_data = pd.read_csv(CFG['test_path'], index_col=False)

    train_dataset = TrainDataset(
        CFG['train_dataset'], 
        train_data,
        train_transforms)
    val_dataset = TrainDataset(
        CFG['val_dataset'],
        val_data,
        val_transforms)
    test_dataset = TestDataset(
        CFG['test_dataset'],
        test_data,
        val_transforms)

    dataloaders: dict = {
        'train': DataLoader(
            dataset=train_dataset,
            batch_size=CFG['bs'],
            num_workers=CFG['num_workers'],
            drop_last=CFG['drop_last'],
            pin_memory=CFG['pin_memory'],
            shuffle=True
            ),
        'val': DataLoader(
            dataset=val_dataset,
            batch_size=CFG['bs'],
            num_workers=CFG['num_workers'],
            drop_last=CFG['drop_last'],
            pin_memory=CFG['pin_memory'],
            shuffle=False,
            )
        }
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=CFG['bs'],
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        drop_last=False,
        shuffle=False,
    )
    if CFG['model'] == 'my':
        model = MyResnet18(BasicBlock, 3, CFG['num_classes'])
    else:
        model = timm.create_model(CFG['model'], pretrained=CFG['pretrain'], num_classes=CFG['num_classes'])
    device = 'cuda:0'
    model = torch.nn.DataParallel(model, device_ids=CFG['device'])
    model = model.to(device)

    if CFG['train']:
        if CFG['loss'] == 'CrossEntropy':
            if CFG['weights']:
                labels = train_data['label']
                unique_labels = np.unique(labels)
                weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
                weights = torch.tensor(weights, dtype=torch.float32)
            else:
                weights=None

            loss = nn.CrossEntropyLoss(weights, label_smoothing=CFG['label_smoothing'])
            loss = loss.to(device)

        if CFG['optim'] == 'Adam':
            optim = Adam(model.parameters(), lr=CFG['lr'])
        elif CFG['optim'] == 'AdamW':
            optim = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        elif CFG['optim'] == 'SGD':
            optim = SGD(model.parameters(), lr=CFG['lr'])

        if CFG['sched'] is None:
            sched = None
        elif CFG['sched'] == 'ReduceLROnPlateau':
            sched = lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.1, patience=10)
        else:
            ...

        metrics = MetricCollection([
            F1Score(num_classes=CFG['num_classes'], average=None).to(device),
            Recall(num_classes=CFG['num_classes'], average=None).to(device),
            Precision(num_classes=CFG['num_classes'], average=None).to(device),
            Accuracy(num_classes=CFG['num_classes'], average=None).to(device)
        ])

        train_loop(
            model=model,
            dataloader=dataloaders,
            loss_func=loss,
            optimizer=optim,
            metric=metrics,
            main_metric=CFG['metric'],
            device=device,
            classes=CFG['num_classes'],
            patience=CFG['patience'],
            scheduler=sched,
            epochs=CFG['epochs'],
            best_model_path=best_model_path,
            last_model_path=last_model_path,
            losses_path=losses_path,
            train_metrics_path=train_metrics_path,
            val_metrics_path=val_metrics_path,
            dataframe_path=dataframe_path
            )
    
    if CFG['test']:
        if CFG['model'] == 'my':
            model = MyResnet18(BasicBlock, 3, CFG['num_classes'])
        else:
            model = timm.create_model(CFG['model'], pretrained=CFG['pretrain'], num_classes=CFG['num_classes'])
        if CFG['use_best']:
            print(f'Start testing, loading model from {best_model_path}')
            model.load_state_dict(torch.load(best_model_path))
        else:
            print(f'Start testing, loading model from {last_model_path}')
            model.load_state_dict(torch.load(last_model_path))

        model = torch.nn.DataParallel(model, device_ids=CFG['device'])
        model = model.to(device)
        test_loop(
            model=model,
            dataloader=test_dataloader,
            device=device,
            submission_path=submission_path
            )


def main():
    CFG = {
        'model': 'timm/pit_b_distilled_224.in1k', #'timm/pit_b_distilled_224.in1k', 'timm/pit_s_distilled_224.in1k' 'timm/convnext_small.fb_in22k_ft_in1k',
        'pretrain': True,
        'loss': 'CrossEntropy',
        'weights': True,
        'label_smoothing': 0.0,
        'optim': 'Adam',
        'lr': 3e-4,
        'weight_decay': 5e-2,
        'sched': 'ReduceLROnPlateau',
        'metric': 'Accuracy',
        'interpolation': cv2.INTER_LINEAR,
        'hf': 0.5,
        'vf': 0.0,
        'sh': 0.0,
        'sc': 0.0,
        'r': 0.0,
        'pd': 0.0,
        'cd': 0.0,
        'imgsz': 224,
        'device': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'bs': 475,
        'num_workers': 4,
        'drop_last': True,
        'pin_memory': False,
        'num_classes': 151,
        'patience': 30,
        'epochs': 100,
        'train': True,
        'test': True,
        'use_best': True,
        'seed': 42,
        'save_path': '/AkhmetzyanovD/projects/MTS/NN/contest_1/results',
        'name': 'default',
        'train_dataset': r'/AkhmetzyanovD/projects/MTS/NN/contest_1/nn-image/train_test_data/train',
        'val_dataset': r'/AkhmetzyanovD/projects/MTS/NN/contest_1/nn-image/train_test_data/train',
        'test_dataset': r'/AkhmetzyanovD/projects/MTS/NN/contest_1/nn-image/train_test_data/test',
        'train_path': r'/AkhmetzyanovD/projects/MTS/NN/contest_1/nn-image/train_val.csv',
        'val_path': r'/AkhmetzyanovD/projects/MTS/NN/contest_1/nn-image/train_val.csv',
        'test_path': r'/AkhmetzyanovD/projects/MTS/NN/contest_1/nn-image/test.csv'
    }
    if not (CFG['train'] or CFG['test']):
        print("CFG['train'] and CFG['test'] is False")
        return None

    train(CFG)


if __name__ == "__main__":
    main()