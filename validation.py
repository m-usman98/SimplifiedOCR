import argparse
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from network.model_arch import Model
import torch.optim as optim
from dataloader import load_dataset
from utils.label_converter import CTCLabelConverter, Averager
from nltk.metrics.distance import edit_distance


def validation(model, evaluation_loader, converter, device, args):
    correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=args.MAX_BATCH_SIZE)

        image = image_tensors.to(device)
        text_for_prediction = torch.LongTensor(batch_size, args.MAX_BATCH_SIZE + 1).fill_(0).to(device)

        start_time = time.time()
        predictions = model(image, text_for_prediction)
        forward_time = time.time() - start_time

        predictions_size = torch.IntTensor([predictions.size(1)] * batch_size)
        cost = criterion(predictions.log_softmax(2).permute(1, 0, 2), text_for_loss, predictions_size, length_for_loss)

        _, predictions_index = predictions.max(2)
        predictions_index = predictions_index.view(-1)
        predictions_str = converter.decode_greedy(predictions_index.data, predictions_size.data)

        infer_time += forward_time
        predictions_prob = F.softmax(predictions, dim=2)
        predictions_max_prob, _ = predictions_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, predictions_str, predictions_max_prob):

            if pred == gt:
                correct += 1
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            confidence_score_list.append(confidence_score)
        valid_loss_avg.add(cost)

    accuracy = correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)
    return valid_loss_avg.val(), accuracy, norm_ED

# return valid_loss_avg.val(), accuracy, norm_ED, predictions_str, confidence_score_list, labels, infer_time, length_of_data


