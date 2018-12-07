# omg_empathy
Repository for the OMG-Empathy Challenge 2019

### Data Processing
To get valence predictions on evey frames and the mutual laughter frame number, extract faces from video and run `python process_data.py` on trainset.

For testset, run `python process_data_trainset.py`. 

Specific the file paths in both scripts:

`path = ./path/to/groundtruth/labels`

`img_path = ./path/to/face/images`

`save_path = ./path/to/save/valence/prediction`

`frame_path = ./path/to/save/mutual/laughter/frame/number`


Todo: instruction on how to get the final csv file?(lets call it final features csv file?)



### Model 1
Run `python train_svm.py`. Specific the file paths in the script:

`path = ./path/to/final/features/csv/files`

`savepath = ./path/to/save/svm/model`


### Model 2
Instruction on how to run bita's model

### Result
| Subject       | Baseline CCC  | Model 1 CCC  |
| ------------- |-------------| -----|
| Subject 1     | 0.01 | 0.59 |
| Subject 2     | 0.11 | 0.15 |
| Subject 3     | 0.04 | 0.50 |
| Subject 4     | 0.1 |  0.22 |
| Subject 5     | 0.11 | 0.28 |
| Subject 6     | 0.35 | 0.30 |
| Subject 7     | -0.01 | -0.16 |
| Subject 8     | 0.05 | -0.01 |
| Subject 9     | 0.05 | 0.11 |
| Subject 10     | 0.10 | -0.02 |
| Mean    | 0.091     |    0.19 |
