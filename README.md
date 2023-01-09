# Tabular Playground Series - Aug 2022


This repository is the ML final project for National Yang Ming Chiao Tung University machine learning course.
[Link](https://docs.google.com/presentation/d/15d4W_8GFks4Mqmf4kvmTxYC8tJv-KNg6c8rQrlccEWM/edit#slide=id.g61dd2f3d9d_2_83) 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model, run `109550032_Final_train.ipynb`.

Or another option is you can train the model with this command.

```train
python3 train.py 
```
##### Hyperparameters:
```python3
epoch = 200
batch_size = 200
loss = binary_crossentropy
optimizer='adam'
```
##### Model Structure:
```python3
model = Sequential()
model.add(Dense(92, input_dim=23))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(92))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```
## Evaluation

To evaluate the model, run `109550032_Final_inference.ipynb`.

Or another opthion is you can evaluation the model with this command.

```eval
python evalution.py
```

The result will be `result.csv`

## Pre-trained Models

You can download pretrained models here:
- [pretrain model](https://drive.google.com/file/d/1qDgQIF_rlINKgHKlQcXX8swuWqbVbFmm/view?usp=sharing)

## Results

Private score -> 0.59134

## Reproduce step 

```
git clone https://github.com/detaomega/ML_project.git
```

Then you can train and evaluate the model.
