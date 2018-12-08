import os
import os.path
import numpy as np
import random
import math
import datetime
import csv

import torch.utils.data
import torchvision.transforms as transforms

class ValenceDataLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, subject_id, trajectory_length=10, mean=0, test=False):
        self.base_path = datapath
        if test:
            self.filename = 's{}.test'.format(subject_id)
        else:
            self.filename = 's{}.data'.format(subject_id)

        self.size = 0
        self.features = []
        self.valences = []
        self.load_data()
        self.trajectory_length = trajectory_length
        self.mean = mean

    def load_data(self):
        self.features = []
        self.valences = []
        with open(os.path.join(self.base_path, self.filename)) as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                self.features.append([
                    float(row[0]), # image valence mean
                    float(row[1]), # emo_watson
                    float(row[2]), # opensmile_valence
                    float(row[3]), # opensmile_arousal
                    float(row[4]), # polarity
                    float(row[5])  # both_laugh
                ])
                ## classifier
                #if (row[6] > 0.1):
                #    self.valences(1)
                #elif (row[6] < -0.1):
                #    self.valences(2)
                #else:
                #    self.valences(0)
                ## regression
                self.valences.append(float(row[6]))

        self.size = len(self.features)

    def __getitem__(self, index):
        features = []
        valences = []
        for i in range(index, index+self.trajectory_length):
            features.append(self.features[i])
            # valences.append((self.valences[i] - self.mean) / (1. - self.mean))
            valences.append(np.clip(self.valences[i] - self.mean, -1., 1.))
        
        return np.asarray(features), np.asarray(valences)
        # return np.asarray(self.features[index]), self.valences[index]

    def __len__(self):
        return self.size - (self.trajectory_length)

if __name__ == "__main__":
    db = ValenceDataLoader(datapath="data/", subject_id=6, trajectory_length=1)
    mean = 0
    for features, valences in db:
        mean += valences
    mean /= len(db)
    print (mean)
