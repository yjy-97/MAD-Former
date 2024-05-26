import os
import sys
import json
import pickle
import random
import math
import logging
import torch
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics

logger = logging.getLogger(__name__)


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(3407)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "not find data for train."
    assert len(val_images_path) > 0, "not find data for eval"

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    global m_8_f, m_16_f
    m_8_f = torch.zeros(2, 2, 8).to(device)
    m_16_f = torch.zeros(2, 2, 12).to(device)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred, m_8, m_16 = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        m_8 = m_8.type_as(m_8_f)
        m_8_cos = torch.cosine_similarity(m_8_f, m_8, dim=2, eps=1e-08).to(device)


        m_8_cos = torch.mean(m_8_cos).to(device)
        m_8_cos = 1 - m_8_cos
        print('m_8_cos', m_8_cos)

        m_16 = m_16.type_as(m_16_f)
        m_16_cos = torch.cosine_similarity(m_16_f, m_16, dim=2, eps=1e-08).to(device)
        m_16_cos = torch.mean(m_16_cos).to(device)
        m_16_cos = 1 - m_16_cos
        print('m_16_cos', m_16_cos)

        m_8_f = m_8
        m_16_f = m_16

        # loss = loss_function(pred, labels.to(device))
        loss = loss_function(pred, labels.to(device)) + 0.2 * m_8_cos + 0.2 * m_16_cos
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    global m_8_f, m_16_f
    m_8_f = torch.zeros(2, 2, 8).to(device)
    m_16_f = torch.zeros(2, 2, 12).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    all_preds, all_label = [], []
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred, m_8, m_16 = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        m_8 = m_8.type_as(m_8_f)
        m_8_cos = torch.cosine_similarity(m_8_f, m_8, dim=2, eps=1e-08).to(device)
        m_8_cos = torch.mean(m_8_cos).to(device)
        m_8_cos = 1 - m_8_cos

        m_16 = m_16.type_as(m_16_f)
        m_16_cos = torch.cosine_similarity(m_16_f, m_16, dim=2, eps=1e-08).to(device)
        m_16_cos = torch.mean(m_16_cos).to(device)
        m_16_cos = 1 - m_16_cos

        m_8_f = m_8
        m_16_f = m_16

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        # loss = loss_function(pred, labels.to(device))
        loss = loss_function(pred, labels.to(device)) + 0.2 * m_8_cos + 0.2 * m_16_cos
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
        if len(all_preds) == 0:
            all_preds.append(pred_classes.detach().cpu().numpy())
            all_label.append(labels.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], pred_classes.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], labels.detach().cpu().numpy(), axis=0
            )

    all_preds, all_label = all_preds[0], all_label[0]
    report = metrics.classification_report(all_label, all_preds, target_names=['0', '1'], digits=4)
    logger.info(report)
    print(report)
    print("AUC:{:.4f}".format(metrics.roc_auc_score(all_label, all_preds)))
    print("F1-Score:{:.4f}".format(f1_score(all_label, all_preds, average='macro')))
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
