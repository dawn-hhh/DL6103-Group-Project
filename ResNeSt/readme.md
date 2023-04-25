These codes implements various models with Cifar100 dataset: 

1. conf/global_settings.py: change Epochs (line 16), milestones (line 17), and output directory (line 24)

2. models/...: contains the network structures for various models

3. dataset.py: creates the cifar100 dataset

4. train.py: script for model training and evaluation
             change batch size/learning rate (line 154/155)
             change optimizer/learning rate scheduler (line 180-187)
             output training curve plot (line 266)

5. train.sh: script for submitting job to GPU cluster 
             change the type of model (line 12)

6. utils.py & utils1.py: contains self-defined functions which are called in train.py
