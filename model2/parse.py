from textblob import TextBlob
import argparse
import io
import csv
import pandas as pd

f = []
filename = raw_input()
print("parsing...")
filetext = filename + ".txt"
sxsx = filename + ".csv")
frames = []
co = pd.read_csv("Frames_Count.txt", delimiter=" - ",  names= ["name", "number"], engine='python') 
for k, row in co.iterrows():
	if ( row['name'] == sxsx ):
		numberofFrames = int(filter(str.isdigit, row["number"]))
indexF = []
for k in range(1, numberofFrames+1):
	indexF.append(k)
'''
g = []
g.append("csv/")
g.append(filename)
g.append(".csv")
filecsv = ''.join(g)
'''
h = []
h.append("csv_valence/")
h.append(filename)
h.append(".csv")
filecsv_valence = ''.join(h)
filewatson = "watson_output/" + filename  + ".csv"
fileopensmile = "opensmile_output/" + filename + ".csv"
t = open(filetext,'r')                                                                  
dfWords = pd.read_csv(sxsx)
dfOpen = pd.read_csv(fileopensmile, names = ["seconds", "valence", "arousal"])
dfWatson = pd.read_csv(filewatson, names = ["value"])
dfFinal = pd.DataFrame(index = indexF)
dfFace = pd.read_csv(filecsv_valence, names = ["value"])
text = t.read()
txt = text.replace("'", "")
blob = TextBlob(txt)
blob.tags         
blob.noun_phrases
listWords = []
words = []
listWords = []
words = []
w = ""
for sentence in blob.sentences:
        pol = sentence.sentiment.polarity
        words = sentence.strip().split(' ')
        for i in range(len(words)):
                w = words[i].replace('.',' ').replace('?',' ').replace(',',' ').replace('!',' ').replace(';',' ').replace(':',' ').lower()
                listPolarity = []
                listPolarity.append(w)
                listPolarity.append(pol)
                listWords.append(listPolarity)

headers =['words','polarity']
dfPolarity = pd. DataFrame(listWords, columns=headers)
dfWords['polarity'] = ""
idxV = 0
for idxT, row in dfPolarity.iterrows():
	if (idxV < dfPolarity.shape[0]):
		dfWords.loc[idxV, 'polarity'] = row['polarity']	
		idxV += 1

dfWords['emo_watson'] = ""
idxV = 0
for idxT, row in dfWatson.iterrows():
	if (idxV < dfWatson.shape[0]):
		dfWords.loc[idxV, 'emo_watson'] = row['value']	
		idxV += 1

dfFinal.index.name = "index"
'''for idx, row in dfFinal.iterrows():
for j in range(len(0,numberofFrames)):
	dfFinal.loc[j, "seconds"] =  (j/25).astype(float)
'''	
dfFinal['seconds']=(((dfFinal.index.values).astype(float))/25).astype(float)
dfFinal['words'] = ""
dfFinal['polarity'] = ""
dfFinal['image_valence'] = ""
dfFinal['emo_watson'] = ""
dfFinal['opensmile_valence'] = ""
dfFinal['opensmile_arousal'] = ""
idxT = 0
idxV = 1
for idxT, row in dfWords.iterrows():
        while(idxV < numberofFrames ):
                if (dfFinal.loc[idxV,'seconds'] < row['end']):
                        dfFinal.loc[idxV, 'words'] = row['words'].lower()
                        dfFinal.loc[idxV, 'polarity'] = row['polarity']
                        dfFinal.loc[idxV, 'emo_watson'] = row['emo_watson']
                        idxV += 1
                else:
                        break
idxS = 1
for idx, row in dfOpen.iterrows():
	while(idxS < numberofFrames):	
                if (dfFinal.loc[idxS,'seconds'] < row['seconds']):
                        dfFinal.loc[idxS, 'opensmile_arousal'] = row['arousal']
                        dfFinal.loc[idxS, 'opensmile_valence'] = row['valence']
			idxS += 1
		else:
			break
sec = 1
idxl = 0
idxf = 1
for idxl, row in dfFace.iterrows():
        while (idxf < numberofFrames):
                if (dfFinal.loc[idxf,'seconds'] < sec):
                        dfFinal.loc[idxf, 'image_valence'] = row['value']
                        idxf += 1

                else:
                        sec += 1
                        dfFinal.loc[idxf, 'image_valence'] = row[0]
                        idxf += 1
                        break;

output = "results/" + filename + ".csv"
dfFinal.to_csv(output)
