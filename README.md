# Ship detection on satellite images using deep learning


This repository contains my research of the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/).

## Goals, description of the competition

We are given satellite images (more accurately sections of satellite images), which might contain ships or other waterborne vehicles.
The goal is to segment the images to the "ship"/"no-ship" classes (label each pixel using these classes).
The images might contain multiple ships, they can be placed close to each other (yet should be detected as separate ships), they can be located in ports, can be moving or stationary, etc.
The pictures might show inland areas,the sea without ships, can be cloudy or foggy, lighting conditions can vary.
The training data is given as images and masks for the ships (in a run length encoded format).
If an image contains multiple ships, each ship has a separate record, mask.

## Motivations

Earth observation using satellite data is a rapidly growing field.
We use satellites to monitor polar ice caps, to detect environmental disasters such as tsunamis, to predict the weather, to monitor the growth of crops and many more.
Shipping traffic is growing fast.
More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement.
This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.

## Prerequisites

The [training data](https://www.kaggle.com/competitions/airbus-ship-detection/data) can be downloaded from the competition's website.
I used Python 3.8 with Keras and Tensorflow and some other necessary packages.

## Data exploration

The training data is analysed and visualised in the [Analysis_of_Airbus_Ship_Detection.ipynb](https://github.com/Surovytskyi/Airbus-Ship-Detection/blob/main/Analysis_of_Airbus_Ship_Detection%20.ipynb) Jupyter Notebook. 

## Getting started

### Clone this repo:

`git clone https://github.com/Surovytskyi/Airbus-Ship-Detection.git`

### Install all necessary requirements:

`pip install -r requirements.txt`

### Specify the path to the dataset in `data_preparing.py` file:

`ship_dir = 'airbus-ship-detection-dataset'`

`train_image_dir = os.path.join(ship_dir, 'train_v2')`

`masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))`

## Training

The `helper_functions.py` script contains helper functions 
and the `train_data_generation.py` script contains generator class which are to run the train script.

Running the script `Train.py` runs `data_preparing.py` script, which loads and preprocesses the training data for a good balances in each set,
it includes network definitions and the training. The model is saved to model.hdf5 after every epoch based on dice score which improves network performance.

## Evaluation

The `helper_functions.py` script contains helper functions 
and the `test_data_generation.py` script contains generator class which are to run the test script.

Once you have a trained model, the script `Evaluation.py` will take the saved model from the training run and use it to predict image masks and their evaluation by the corresponding metrics.

## Fine-tuning

The `options.py` script contains all the parameters for fine-tuning and parsing training/test script arguments.
