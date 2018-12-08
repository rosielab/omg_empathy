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

path = '../../OMGEmpathyChallenge-master/data/testfile/'
savepath = '../../OMGEmpathyChallenge-master/data/test_result/'
svmpath = './svm/'
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
    for k in range(len(subjectlist)):
        poltest = []
        poltest2 = []
        predtest = []
        fn = subjectlist[k]
        print('== PROCESSING {} =='.format(fn))
        file = path + '{}.csv'.format(fn)
        dataset = read_csv(file, header=0, index_col=0)
        values = dataset.values
        a = values[:,4].tolist()
        #b = values[:,0].tolist()
        c = values[:,3].tolist()
        d = values[:,2].tolist()
        #print(len(a))
        for i in range(len(a)):
            poltest.append(a[i])
            #groundtruth.append(b[i])
            predtest.append(c[i])
            poltest2.append(d[i])
        #print(len(poltest))
        valence = []
        i = 0    
        poltestno = [x for x in poltest if str(x) != 'nan']
        clf = load(svmpath + '{}.joblib'.format(z)) 
        while i < len(poltest): 
            if z in [7,8]:
                if poltest2[i] == poltest2[i]:
                    if poltest2[i] != 0 or i ==0:
                        v = (clf.predict([[(predtest[i]),(poltest2[i])]])[0])
                    else:
                        v = valence[i-1]
                        if z == 8:
                            v = 0
                else:
                    v = valence[i-1] 
            else:
                if poltest[i] == poltest[i]:
                    if poltest[i] != 0 or i == 0:
                        v = (clf.predict([[(predtest[i]),(poltest[i])]])[0])
                    else:
                        v = valence[i-1]
                    if z in [1,2,3,5,6]:
                        if poltest[i] > (sum(poltestno)/len(poltestno)):
                            v = abs(predtest[i])
                else:
                    v = valence[i-1]
            valence.append(v)
            i = i + 1
        #print(len(valence))
        #print(len(poltest))
        savefile = savepath + '{}.csv'.format(fn)
        with open(savefile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(['valence'])
            for val in valence:
                writer.writerow([val])
        
print('FINISHED')

