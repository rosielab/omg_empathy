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


Todo: instruction on how to get the final csv file?(lets call it final features csv file)



### Model 1
Run `python train_svm.py`. Specific the file paths in the script:

`path = ./path/to/final/features/csv/files`

`savepath = ./path/to/save/svm/model`


### Model 2
Instruction on how to run bita's model
