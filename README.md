# Image-Retrieval-with-Generalized-Mean-Descriptor
A simple content-based image retrieval system.

![](https://raw.githubusercontent.com/runjia0124/Image-Retrieval-with-Generalized-Mean-Descriptor/main/archive/retrieval.png)

## What we use
- Generalized Mean Descriptor (TPAMI'18)
- resnet50 + whitening 
- LSHash


## Usage

To test the algorithm:
- put your dataset into `./database/data`
- run `python setup.py` to generate `feature` and `index`

To test this simple interface:
- run `python interface.py`
- pack this code as an executable file if you like 

## Reach me

junko.lin@yahoo.com

We express our great thanks to the following repos:  
https://github.com/filipradenovic/cnnimageretrieval-pytorch  
https://github.com/kayzhu/LSHash  
