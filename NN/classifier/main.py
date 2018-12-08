import io
import os
from PIL import Image
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import csv

from scipy.stats import pearsonr
import pandas

import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from valencenet import ValenceNet
from dataset import ValenceDataLoader
epoch = 0
USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
K = 100.

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def train_model(model, train_loader, criterion, optimizer, epoch, batch_size):
    model.train()
    model.training = True

    # switch to train mode
    total = 0
    running_loss = 0
    running_corrects = 0
    for batch_idx, (features_stacked, labels, valences) in enumerate(train_loader):
        if USE_CUDA:
            features_stacked, labels = features_stacked.cuda(), labels.cuda()
        features_stacked, labels = Variable(features_stacked).type(FLOAT), Variable(labels)

        estimated_labels = model(features_stacked)
        loss = criterion(estimated_labels, labels)
        _, predicted_labels = torch.max(estimated_labels.data, 1)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(predicted_labels == labels.data)
        total += labels.size(0)

    epoch += 1
    print('-----------------------------------------------------------------------------')
    print(epoch)
    epoch_loss = (float(running_loss) / float(total)) * 100.0
    epoch_acc = (float(running_corrects) / float(total)) * 100.0
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
           'train', epoch_loss, epoch_acc))

def train(model, datapath, subject_id, checkpoint_path, epochs, args):
    counts = [0, 0, 0]
    db = ValenceDataLoader(datapath=datapath, subject_id=subject_id)
    for _, labels, _ in db:
        if (labels == 0):
            counts[0] += 1;
        elif (labels == 1):
            counts[1] += 1;
        elif (labels == 2):
            counts[2] += 1;

    print (counts)
    min_count = float(min(counts))
    class_weights = [min_count / counts[0], min_count / counts[1], min_count / counts[2]]
    print (class_weights)
    class_weights = FLOAT(class_weights)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(ValenceDataLoader(datapath, subject_id=subject_id), batch_size=args.bsize, shuffle=True, drop_last=True, **kwargs)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    eval_loader = torch.utils.data.DataLoader(ValenceDataLoader(datapath, subject_id=subject_id, test=True), batch_size=args.bsize, shuffle=False, **kwargs)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_CCC = 0.
    for epoch in range(1, epochs + 1):
        # train for one epoch
        train_model(model, train_loader, criterion, optimizer, epoch, args.bsize)
#        # evaluate on validation set
        CCC = eval_model(model, eval_loader, args.bsize)
        if (CCC is None):
            CCC = 0.

#        # remember best acc and save checkpoint
        is_best = CCC > best_CCC
        best_CCC = max(CCC, best_CCC)
        if (is_best):
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_{}.pth".format(epoch)))

def eval_model(model, eval_loader, batch_size):
    model.eval()
    model.training = False

    total = 0
    running_corrects = 0
    predictions = []
    probabilities = []
    ground_truth = []
    ground_truth_valences = []
    features = []
    for batch_idx, (features_stacked, labels, valences) in enumerate(eval_loader):
        if USE_CUDA:
            features_stacked, labels = features_stacked.cuda(), labels.cuda()
        features_stacked, labels = Variable(features_stacked).type(FLOAT), Variable(labels)
        estimated_labels = model(features_stacked)
        predicted_probabilities, predicted_labels = torch.max(estimated_labels.data, 1)
        # print (predicted_labels)
        for i in range(labels.size(0)):
            predictions.append(predicted_labels[i].item())
            ground_truth.append(labels[i].item())
            ground_truth_valences.append(valences[i].item())
            features.append(features_stacked[0].data.cpu().numpy())
            probabilities.append(predicted_probabilities[i].item())

        running_corrects += torch.sum(predicted_labels == labels.data)
        total += labels.size(0)

    predictions_out = []
    for i in range(len(predictions)):
        if predictions[i] == 1:
            predictions_out.append(1.)
            # predictions_out.append(abs(probabilities[i]))
        elif predictions[i] == 0:
            predictions_out.append(0)
        else:
            predictions_out.append(-1.)
            # predictions_out.append(-abs(probabilities[i]))
    ground_truth_out = []
    for label in ground_truth:
        if label == 1:
            ground_truth_out.append(1.)
        elif label == 0:
            ground_truth_out.append(.0)
        else:
            ground_truth_out.append(-1.)

    # predictions = [-0.5 if label == 2 else float(label) for label in predictions]
    # ground_truth = [-0.5 if label == 2 else float(label) for label in ground_truth]

    epoch_acc = (float(running_corrects) / float(total)) * 100.0
    CCC, _ = ccc(ground_truth_valences, predictions_out)
    print('{} CCC: {:.4f} Acc: {:.4f}'.format(
           'test', CCC, epoch_acc))

    with open('output.csv','w') as f:
         writer = csv.writer(f)
         predictions_2d = [[prediction] for prediction in predictions_out]
         writer.writerows(predictions_2d)

    features = np.asarray(features)
    plt.clf()
    idx = range(len(predictions_out))
    plt.plot(idx, predictions_out, label = 'predictions', alpha = 0.7)
    plt.plot(idx, ground_truth_out, label = 'ground_truth', alpha = 0.7)
    plt.plot(idx, ground_truth_valences, label = 'ground_truth_valences', alpha = 0.5)
#    plt.plot(idx, probabilities, label = 'probabilties', alpha = 0.5)
#    plt.plot(idx, features[:,0], label = 'image_valence_mean', alpha = 0.2)
#    plt.plot(idx, features[:,1], label = 'emo_watson', alpha = 0.2)
#    plt.plot(idx, features[:,2], label = 'opensmile_valence', alpha = 0.2)
#    plt.plot(idx, features[:,3], label = 'opensmile_arousal', alpha = 0.2)
#    plt.plot(idx, features[:,4], label = 'polarity', alpha = 0.2)
#    plt.plot(idx, features[:,5], label = 'both_laugh', alpha = 0.2)
    plt.legend()
    plt.pause(0.05)
    return CCC

def eval(model, datapath, subject_id):
    model.eval()
    model.training = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    eval_loader = torch.utils.data.DataLoader(ValenceDataLoader(datapath, subject_id=subject_id, test=True), batch_size=32, shuffle=False, **kwargs)
    return eval_model(model, eval_loader, 32)

def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true,y_pred)

def f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    label = [0,1,2,3,4,5,6]
    return f1_score(y_true,y_pred,labels=label,average="micro")

def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = np.std(y_pred)

    std_gt = np.std(y_true)

    ccc = 2 * rho * std_gt * std_predictions / (std_predictions ** 2 + std_gt ** 2 + (pred_mean - true_mean) ** 2)

    return ccc, rho

def test_model(model, eval_loader, batch_size, subject_id, story_id):
    model.eval()
    model.training = False

    total = 0
    running_corrects = 0
    predictions = []
    probabilities = []
    ground_truth = []
    ground_truth_valences = []
    features = []
    for batch_idx, (features_stacked, labels, valences) in enumerate(eval_loader):
        if USE_CUDA:
            features_stacked, labels = features_stacked.cuda(), labels.cuda()
        features_stacked, labels = Variable(features_stacked).type(FLOAT), Variable(labels)
        estimated_labels = model(features_stacked)
        predicted_probabilities, predicted_labels = torch.max(estimated_labels.data, 1)
        # print (predicted_labels)
        total += labels.size(0)
        for i in range(labels.size(0)):
            predictions.append(predicted_labels[i].item())
            features.append(features_stacked[0].data.cpu().numpy())
            probabilities.append(predicted_probabilities[i].item())

    predictions_out = []
    for i in range(len(predictions)):
        if predictions[i] == 1:
            predictions_out.append(1.)
            # predictions_out.append(abs(probabilities[i]))
        elif predictions[i] == 0:
            predictions_out.append(0)
        else:
            predictions_out.append(-1.)
            # predictions_out.append(-abs(probabilities[i]))

    output_filename = "Subject_{}_Story_{}_out.csv".format(subject_id, story_id)
    with open(output_filename,'w') as f:
         writer = csv.writer(f)
         predictions_2d = [[prediction] for prediction in predictions_out]
         writer.writerows(predictions_2d)

    features = np.asarray(features)
    plt.clf()
    idx = range(len(predictions_out))
    plt.plot(idx, predictions_out, label = 'predictions', alpha = 0.7)
    # plt.plot(idx, probabilities, label = 'probabilties', alpha = 0.5)
    plt.plot(idx, features[:,0], label = 'image_valence_mean', alpha = 0.2)
    plt.plot(idx, features[:,1], label = 'emo_watson', alpha = 0.2)
    plt.plot(idx, features[:,2], label = 'opensmile_valence', alpha = 0.2)
    plt.plot(idx, features[:,3], label = 'opensmile_arousal', alpha = 0.2)
    plt.plot(idx, features[:,4], label = 'polarity', alpha = 0.2)
    plt.plot(idx, features[:,5], label = 'both_laugh', alpha = 0.2)
    plt.legend()
    plt.pause(0.05)

def test(model, datapath, subject_id, story_id):
    model.eval()
    model.training = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_loader = torch.utils.data.DataLoader(ValenceDataLoader(datapath, subject_id=subject_id, test=True, story_id=story_id), batch_size=32, shuffle=False, **kwargs)
    return test_model(model, test_loader, 32, subject_id, story_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch on Place Recognition + Visual Odometry')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/eval/test')
    parser.add_argument('--subject', default=1, type=int, help='subject_id')
    parser.add_argument('--story', default=1, type=int, help='story_id')
    parser.add_argument('--datapath', default='data', type=str, help='path KITII odometry dataset')
    parser.add_argument('--bsize', default=1, type=int, help='minibatch size')
    parser.add_argument('--trajectory_length', default=10, type=int, help='trajectory length')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_iter', default=20000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--checkpoint_path', default='checkpoints/', type=str, help='Checkpoint path')
    parser.add_argument('--checkpoint', default=None, type=str, help='Checkpoint')
    args = parser.parse_args()

    model = ValenceNet()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    if USE_CUDA:
        model.cuda()

    args = parser.parse_args()
    if args.mode == 'train':
        train(model, args.datapath, args.subject, args.checkpoint_path, args.train_iter, args)
    elif args.mode == 'eval':
        eval(model, args.datapath, args.subject)
        plt.show()
    elif args.mode == 'test':
        test(model, args.datapath, args.subject, args.story)
        plt.show()
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
