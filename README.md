# Finding Tori: Self-supervised Learning for Analyzing Korean Folk Song (ISMIR 2023)

This is the official code repository for our ISMIR 2023 paper.

[Paper]() \
[Visualization demo](https://danbinaerinhan.github.io/korean-folksong-visualization/)

----
## Anthology of Korean Traditional Folksongs
[Official website](http://urisori.co.kr/urisori-en/doku.php?id=start)

The metadata we collected from the website is available in `metadata.csv`, including the url for each song. You can download the audio files from the website.

## Contour Dataset
We used [CREPE](https://github.com/marl/crepe) to extract the F0 contours from the audio dataset. You can download pre-processed CSV files from the google drive link

```
gdown gdown 18G6QaIqruyQBdJeMX9bjBasi5PFDuxh-
unzip finding_tori_contour_csv.zip
```

This will make `contour_csv/` directory that includes csv files.


## Requirements
### Pipenv Environment
We used pipenv to manage the python environment. You can install the environment by running the following command.

```
pip install pipenv
pipenv install
pipenv shell
```

### (Python Packages)
If you don't want to use pipenv, you can install the packages that are listed in `requirements.txt`. You can install them by running the following command.

```
pip install -r requirements.txt
```




## Reproduce the Result
You can get the result of table 1 in our paper by running ``python3 get_eval_result.py``. For CNN models, pre-trained weights are available in `pretrained_weights/` directory. 


  - The script gives nDCG and RF classifier accuracy in two results. Including `others` class or not.
  - To get result of Pitch Histogram with 25 bins, run ``python3 get_eval_result.py --use_histogram --resolution=1``
  - To get result of Pitch Histogram with 124 bins, run ``python3 get_eval_result.py --use_histogram --resolution=0.2``
  - To get result of Region-supervised CNN model, run ``python3 get_eval_result.py --model=region-trained``
  - To get result of Self-supervised CNN model, run ``python3 get_eval_result.py --model=self-supervised``
  - If you want to use your own trained model, you can specify the path of the model by ``--ckpt_path=path_to_model_state.pt`` option.

## Train the model
You can also train the CNN model again with CSV datasets. 

```
python3 train.py                          # train the self-supervised model
python3 train.py exp="terrain_classifier" # train the region-supervised model
``` 

The `train.py` uses [hydra](https://hydra.cc/docs/intro/) package to handle configuration. The checkpoint will be saved in `./experiments_checkpoints/`

After the training, you can get the evaluation result by running code below
```
python3 get_eval_result.py --ckpt_path=experiments_checkpoints/{date_time_model_code}/model_state.pt
```