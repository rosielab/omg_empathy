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
    def __init__(self, datapath, subject_id, test=False, story_id=None):
        self.base_path = datapath
        if test:
            if (story_id is None):
                self.filename = 's{}.test'.format(subject_id)
            else:
                self.filename = 'Subject_{}_Story_{}_c50.csv'.format(subject_id, story_id)
        else:
            self.filename = 's{}.data'.format(subject_id)

        self.is_test = test
        self.size = 0
        self.features = []
        self.valences = []
        self.labels = []
        if (story_id is None):
            self.load_data()
        else:
            self.load_data_test()

    def load_data_test(self):
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
                ## regression
                self.valences.append(0.)
                self.labels.append(0)
        self.size = len(self.features)

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
                ## regression
                self.valences.append(float(row[6]))
                ## classifier
                if (float(row[6]) > 0.2): # 0.2
                    self.labels.append(1)
                elif (float(row[6]) < -0.2): # 0.2
                    self.labels.append(2)
                else:
                    self.labels.append(0)

        """
        ## calculate mean
        if (self.is_test == False):
           valence_mean = np.mean(self.valences)
           self.valences = [valence - valence_mean for valence in self.valences]
        """

        """
        mean_positive = 0.
        counter_positive = 0
        mean_negative = 0.
        counter_negative = 0
        for valence in self.valences:
            if (valence > 0.):
                mean_positive += valence
                counter_positive += 1.
            elif (valence < 0.):
                mean_negative += valence
                counter_negative += 1.
        mean_positive /= counter_positive
        mean_negative /= counter_negative

        ## find the threshold for classes
        for i in range(len(self.valences)):
            if (self.valences[i] > mean_positive):
                self.labels[i] = 1
            elif (self.valences[i] < mean_negative):
                self.labels[i] = 2
            else:
                self.labels[i] = 0
        """

        if (self.is_test == False):
            valences_randomized = [valence + random.uniform(-0.001, 0.001) for valence in self.valences]
        else:
            valences_randomized = [valence for valence in self.valences]
        # valences_randomized = [valence for valence in self.valences]
        thresholds = np.percentile(np.array(valences_randomized), [100.0/3, 200./3])
        print (thresholds)
        for i in range(len(self.valences)):
            if (valences_randomized[i] < thresholds[0]):
                self.labels[i] = 2
            elif (valences_randomized[i] > thresholds[1]):
                self.labels[i] = 1
            else:
                self.labels[i] = 0

        self.size = len(self.features)

    def __getitem__(self, index):
        return np.asarray(self.features[index]), self.labels[index], self.valences[index]

        # features = []
        # for i in range(-9, 1):
        #     idx = max(index+i, 0)
        #     features = np.concatenate([features, self.features[idx]])
        # return np.asarray(features), self.labels[index], self.valences[index]

    def __len__(self):
        return self.size

if __name__ == "__main__":
    db = ValenceDataLoader(datapath="data/multiple_both", subject_id=6, trajectory_length=1)
    mean = 0
    for features, valences in db:
        mean += valences
    mean /= len(db)
    print (mean)
