# omg_empathy
Code for paper: EmoCog Model for Multimodal Empathy Prediction

Submission to the OMG-Empathy Challenge 2019

### Data Processing
To get valence predictions on evey frames and the mutual laughter frame number, extract faces from video and run `python process_data.py` on trainset. For testset, run `python process_data_trainset.py`. 

Specific the file paths in both scripts:

`path = ./path/to/groundtruth/labels`

`img_path = ./path/to/face/images`

`save_path = ./path/to/save/valence/prediction`

`frame_path = ./path/to/save/mutual/laughter/frame/number`


Todo: instruction on how to get the final csv file?(lets call it final features csv file?)



### Model 1
To train the model, run `python train_svm.py`. Specific the file paths in the script:

`path = ./path/to/final/features/csv/files(trainset)`

`savepath = ./path/to/save/svm/model`

To test the model, run `python test_svm.py`. Specific the file paths in the script:

`path = ./path/to/final/features/csv/files(testset`

`savepath = ./path/to/save/test/results`

`svmpath = ./path/to/saved/svm/models`


### Model 2
* [non-verbal features]
	- openSmile feature extraction tool [https://www.audeering.com/technology/opensmile/]
	- Emosic [https://arxiv.org/abs/1807.08775]
* [verbal features] 
	- tone analyzer (IBM watson) [https://tone-analyzer-demo.ng.bluemix.net/]	
	- TextBlob python API [https://textblob.readthedocs.io/en/dev/]

The audio of each video extracted (using ffmpeg[https://www.ffmpeg.org/]).
`Speech` folder consists of the text of each video alongside a csv file with time offset values (timestamps) for the beginning and end of each spoken word, using Speech-to-text Google API [https://cloud.google.com/speech-to-text/]

`$ python src/parse.py `
input desired file `Subject_X_Story_X`


### Result
**Test results on validation set:**

| Subject       | Baseline CCC  | Model 1 CCC  | Model 2 CCC |
| ------------- |-------------| -----|-----|
| Subject 1     | 0.01 | 0.59 | |
| Subject 2     | 0.11 | 0.15 | |
| Subject 3     | 0.04 | 0.50 | |
| Subject 4     | 0.1 |  0.22 | |
| Subject 5     | 0.11 | 0.28 | |
| Subject 6     | 0.35 | 0.30 | |
| Subject 7     | -0.01 | -0.16 | |
| Subject 8     | 0.05 | -0.01 | |
| Subject 9     | 0.05 | 0.11 | |
| Subject 10     | 0.10 | -0.02 | |
| Mean    | 0.091     |    0.19 | |

**Test results using five fold cross validation:**

| Subject       | Model 1 CCC  | Model 2 CCC |
| ------------- |-------------| -------------|
| Subject 1     | 0.21 | |
| Subject 2     | 0.20 | |
| Subject 3     | 0.27 | |
| Subject 4     | 0.19 | |
| Subject 5     | 0.21 | |
| Subject 6     | 0.08 | |
| Subject 7     | 0.10 | |
| Subject 8     | 0.06 | |
| Subject 9     | 0.09 | |
| Subject 10     | 0.21 | |
| Mean    | 0.16     | |
