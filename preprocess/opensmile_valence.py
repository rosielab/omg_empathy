import subprocess
import re
import csv
import sys

filename = raw_input()
filepath = "wav/" + filename + ".wav"
output = "../opensmile_output/" + filename + ".csv"

# p = subprocess.Popen('./SMILExtract -C config/emobase_live4_batch.conf -I ~/OMG_project/histories/wav/Subject_6_Story_1.wav', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
p = subprocess.Popen('pushd .; cd /Users/azari/OMG_project/opensmile; ./SMILExtract -C /Users/azari/OMG_project/opensmile/config/emobase_live4_batch.conf -I {}; popd'.format(filepath), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

data = []
timestamp = 0
valence = 0
arousal = 0
for line in p.stdout.readlines():
# 	print (line)
	if (line.find("arousal") != -1):
		result = re.search('~~>(.*)<~~', line)
		if (result is not None):
			arousal = (float(result.group(1)))
	elif (line.find("valence") != -1):
		result = re.search('@ time: (.*),', line)
		if (result is not None):
			timestamp = (float(result.group(1)))
		result = re.search('~~>(.*)<~~', line)
		if (result is not None):
			valence = (float(result.group(1)))
			data.append([timestamp, valence, arousal])


#	print line,
# print (data)
if (len(data) > 0):
	print ("writing {} records to file {}".format(len(data), output))
	resultFile = open(output, 'w')
	wr = csv.writer(resultFile, delimiter=",")
	wr.writerows(data)
else:
	print ("no record to write")

# retval = p.wait()
