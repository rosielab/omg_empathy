# omg_empathy
Repository for the OMG-Empathy Challenge 2019

### Data Processing
To get valence predictions on evey frames and the mutual laughter frame number, extract faces from video and run `python process_data.py` on trainset.

For testset, run `python process_data_trainset.py`. 

Specific the file path in both scripts:

`path = where you saved your groundtruth labels`
`img_path = where you saved your face images`
`save_path = where you want to save the valence prediction`
`frame_path = where you want to save the mutual laughter frame number`


Todo: instruction on how to get the final csv file?



### Model 1
Run `python train_svm.py`.

### Model 2
Instruction on how to run bita's model
