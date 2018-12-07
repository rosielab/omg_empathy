import os
import csv
import numpy as np
#from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import DepthwiseConv2D
from keras.models import load_model
from keras.preprocessing import image
IMAGE_SIZE = 128

def get_regressor_predictions(model, paths):
    valence = []
    for i, path in enumerate(paths):
        img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = image.img_to_array(img) / 255
        img = np.array(img).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        p = model.predict(img)
        valence.append(p[0][0])
    return valence_p

def get_classifier_predictions(model, paths):
    H_r= [] #matching matrix I am looking for
    for i, path in enumerate(paths):
        img = image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = image.img_to_array(img) / 255
        img = np.array(img).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        p = model.predict(img)
        H_r.append(p[0, 1]) #p[0,1] for happy and p[0,0] fo neutral
    return H_r


def get_name (folder):
    #get file name
    dataList = os.listdir(folder)
    name_list = []
    for i in range(len(dataList)):
        name_list.append(dataList[i].split('.')[0])
    return name_list


path = '../../OMGEmpathyChallenge-master/data/labels/Training/'
img_path = '../../OMGEmpathyChallenge-master/data/faces/Training/'
#save_path = '../../OMGEmpathyChallenge-master/data/prediction/Validation/'
save_path = '../../OMGEmpathyChallenge-master/data/temp/'
frame_path = '../../OMGEmpathyChallenge-master/data/temp/dump/'
file_name = get_name(path)
r_path = './M_VGG/R_OUT.h5'
c_path = './M_VGG/C_OUT.h5'
count = 0
for i in range(len(file_name)):
    filename = file_name[i]
    if filename == '':
        pass 
    else: 
        count = count + 1
        print('== COUNT {}=='.format(count))
        print('== PROCESSING {} =='.format(filename))
        label_file = path + '{}.csv'.format(filename)
        img_folder = img_path + '{}.mp4'.format(filename)
        predfile = save_path + '{}.csv'.format(filename)
        framefile = frame_path + '{}.csv'.format(filename)
        truth = []
        frame = []
        with open(label_file) as f:
            reader = csv.reader(f)
            rowNr = 0
            fr = 0
            for row in reader:
                if rowNr >= 1:
                    tmp = float(row[0])
                    truth.append(tmp)
                    fr = fr + 1
                    frame.append(fr)
                rowNr = rowNr + 1
        v_paths = []
        for i in range(0,len(frame)):
            if i % 25 == 0:
                ipath = img_folder + '/Subject/{}.png'.format(i)
                v_paths.append(ipath)
        #print('== CALCULATING ==')
        model = load_model(r_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2D})
        valence_p, arousal_p = get_regressor_predictions(model, v_paths)
        #print('== SAVING RESULTS ==')
        with open(predfile, "w") as output:
            row = 0
            writer = csv.writer(output, lineterminator='\n')
            for val in valence_p:
            #for val in arousal_p:
                writer.writerow([val])
                row = row + 1
        #print('== Done ==')
        #print('== Matching happy face ==')
        v_paths_a = []
        for i in range(0,len(frame)):
            if i % 25 == 0:
                ipath = img_folder + '/Actor/{}.png'.format(i)
                v_paths_a.append(ipath)
        #print('== CALCULATING FIRST ONE ==')
        model = load_model(c_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2D})
        pred_l, pred_r, h, H_r_a = get_classifier_predictions(model, v_paths_a)
        #print('== CALCULATING SECOND ONE==')
        pred_l, pred_r, h, H_r_s = get_classifier_predictions(model, v_paths)
        match_r = []
        for i in range(0,len(H_r_s)):
            ar_t = H_r_a[i]
            sr_t = H_r_s[i]
            if ar_t > 0.25 and sr_t > 0.25:
                match_r.append(1)
            else:
                match_r.append(0)
        matched_frame = []
        for i in range(0,len(match_r)):
            if match_r[i] == 1:
                matched_frame.append(i)
        #print('== SAVING MATCHED FRAMES TO CSV FILE ==')
        with open(framefile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in matched_frame:
                writer.writerow([val])
        #print('== NEXT FILE ==')
print('== FINISHED ==')

