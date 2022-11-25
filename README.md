# Diabetic Retinopathy Detection with Deep Learning

![sample](https://i.imgur.com/Mj1psfA.png)

## Project summary


Diabetic retinopathy (DR) is one of the leading causes of vision loss. Early detection and treatment are crucial steps towards preventing DR. This project considers DR detection as an ordinal classification task and aims at developing a deep learning model for predicting the severity of DR disease based on the patient's retina photograph.

The main data set can be downloaded [APTOS 2019 Blindness Detection competition](https://www.kaggle.com/c/aptos2019-blindness-detection/data). The supplementary data is available [2015 Blindness Detection competition](https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized).
 File `report.pdf` contains a detailed PDF description of the project pipeline.
## Project structure

The project has the following structure:
- `codes/`: codes with modules and functions implementing preprocessing, datasets, model and utilities.
- `notebook`: notebook covering different project stages: data preparation, modeling and ensembling.
- `models/`: model weights saved during training.
- `figures/`: figures exported from the notebooks during the data preprocessing and training.



## There are three stages:
- data exploration and visualization
- pre-training the CNN model on the supplementary 2015 data set
- fine-tuning the CNN model on the main 2019 data set


More details are provided within the notebook and presentation


## Requirements

To run the project codes, you can create a new virtual environment in `conda`:

```
conda create -n diabetic python=3.7
conda activate diabetic
```

and then install the requirements:

- pytorch (torch, torchvision)
- efficientnet-pytorch (pre-trained model weights)
- cv2 (image preprocessing library)
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
