from __future__ import print_function
import json
from os.path import join, dirname
from watson_developer_cloud import ToneAnalyzerV3
from watson_developer_cloud.tone_analyzer_v3 import ToneInput
from textblob import TextBlob
import pandas as pd
import csv

# If service instance provides API key authentication
service = ToneAnalyzerV3(
#     ## url is optional, and defaults to the URL below. Use the correct URL for your region.
    url='https://gateway.watsonplatform.net/tone-analyzer/api',
    version='2017-09-21',
    iam_apikey='H7sXUX39LzgVDqO4IcZaoX3dbsiZYn8GVo2wnZvXd6oP')

filename = raw_input()
f = []
filetext = filename + ".txt"
t = open(filetext,'r')
text = t.read()
tone_input = ToneInput(text)
tone = service.tone(tone_input=tone_input, content_type="application/json")
#print(json.dumps(tone.get_result(), indent=2))
data = tone.get_result()
score1 = 0
score2 = 0


word_valence = []
for i in range(len(data["sentences_tone"])):
	txt = data["sentences_tone"][i]["text"]
	txt = txt.replace("'", "")
	txt = txt.replace(",", "")

	corresponding_valence = 0
	max_probable_tone_id = ''
	max_probablity = -1
	for tone in data["sentences_tone"][i]["tones"]:
		if (tone['tone_id'] in ['anger', 'fear', 'sadness']):
			if (tone['score'] > max_probablity):
				max_probablity = tone['score']
				max_probable_tone_id = tone['tone_id']
				corresponding_valence = -1 * max_probablity
		elif (tone['tone_id'] in ['joy']):
			if (tone['score'] > max_probablity):
				max_probablity = tone['score']
				max_probable_tone_id = tone['tone_id']
				corresponding_valence = max_probablity

	for word in txt.split(' '):
		if (len(word) > 0):
			word_valence.append([word, corresponding_valence])
output = "../watson_output/" + filename + ".csv"
if (len(word_valence) > 0):
	print ("writing {} records to file {}".format(len(word_valence), output))
	resultFile = open(output, 'w')
	wr = csv.writer(resultFile, delimiter=",")
	wr.writerows(word_valence)
else:
	print ("no record to write")
