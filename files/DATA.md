### Data Preparation

#### 3RScan
Please make sure you agree the [3RScan Terms of Use](https://forms.gle/NvL5dvB4tSFrHfQH6) first, and get the download script and put it right at the 3RScan main directory.
Then run
```
cd data
cd 3rscan
# prepare ground truth
bash preparation.sh
```

#### Replica
Download the Replica RGB-D scan dataset using the downloading [script](https://github.com/cvg/nice-slam/blob/master/scripts/download_replica.sh) in Nice-SLAM. It contains rendered trajectories using the mesh models provided by the original Replica datasets.
