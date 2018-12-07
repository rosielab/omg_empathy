import os
from pandas import read_csv
import numpy as np
from sklearn import svm
import csv
from joblib import dump, load



def get_name (folder):
    #get file name
    dataList = os.listdir(folder)
    name_list = []
    for i in range(len(dataList)):
        name_list.append(dataList[i].split('.')[0])
    return name_list

path = '../../OMGEmpathyChallenge-master/data/results/'
savepath = '../../OMGEmpathyChallenge-master/data/svm/'
file_name = get_name(path)
Subject1 = []
Subject2 = []
Subject3 = []
Subject4 = []
Subject5 = []
Subject6 = []
Subject7 = []
Subject8 = []
Subject9 = []
Subject10 = []
for i in range(len(file_name)):
    if file_name[i] == '':
        pass 
    else: 
        s_n = file_name[i].split('_')[1]
        if s_n == '1':
            Subject1.append(file_name[i])
        if s_n == '2':
            Subject2.append(file_name[i])
        if s_n == '3':
            Subject3.append(file_name[i])
        if s_n == '4':
            Subject4.append(file_name[i])
        if s_n == '5':
            Subject5.append(file_name[i])
        if s_n == '6':
            Subject6.append(file_name[i])
        if s_n == '7':
            Subject7.append(file_name[i])
        if s_n == '8':
            Subject8.append(file_name[i])
        if s_n == '9':
            Subject9.append(file_name[i])
        if s_n == '10':
            Subject10.append(file_name[i])
        
subject_final = [Subject1,Subject2,Subject3,Subject4,Subject5,Subject6,Subject7,Subject8,Subject9,Subject10]


for z in range(len(subject_final)):
    subject = subject_final[z]
    subjectlist = subject.copy()
    polarity = []
    polarity2 = []
    groundtruth = []
    pred = []
    for k in range(len(subjectlist)):
        fn = subjectlist[k]
        file = path + '{}.csv'.format(fn)
        dataset = read_csv(file, header=0, index_col=0)
        values = dataset.values
        a = values[:,5].tolist()
        b = values[:,0].tolist()
        c = values[:,4].tolist()
        d = values[:,3].tolist()
        for i in range(0,len(a)):
            polarity.append(a[i])
            groundtruth.append(b[i])
            pred.append(c[i])
            polarity2.append(d[i])
    if z in [7,8]:
        valtest = []
        gtruth = []
        for i in range(len(pred)):
            if polarity2[i] != 0:
                if groundtruth[i] != 0:
                    if pred[i] == pred[i]:
                        if polarity2[i] == polarity2[i]:
                            valtest.append([(pred[i]),(polarity2[i])])
                            gtruth.append(groundtruth[i])

        clf = svm.SVR()
        clf.fit(valtest, gtruth) 
        dump(clf, savepath + '{}.joblib'.format(z)) 
    else:
        valtest = []
        gtruth = []
        for i in range(len(pred)):
            if polarity[i] != 0:
                if groundtruth[i] != 0:
                    if pred[i] == pred[i]:
                        if polarity[i] == polarity[i]:
                            valtest.append([(pred[i]),(polarity[i])])
                            gtruth.append(groundtruth[i])

        clf = svm.SVR()
        clf.fit(valtest, gtruth)
        dump(clf, savepath + '{}.joblib'.format(z)) 
        
print('FINISHED')

