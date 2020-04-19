# Kaggle-Bengali.AI-Handwritten-Grapheme-Classification 167th Place Solution

## Dependencies
pytorch == 1.4.0 <br />
cudatoolkit == 10.1 <br />
[efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch) <br />
numpy == 1.18.1 <br />
pandas == 1.0.3 <br />
scikit-learn == 0.22.1 <br />
apex == 0.1 <br />
tqdm == 4.44.1 <br />

## Solution
https://www.kaggle.com/c/bengaliai-cv19/discussion/144549

## Dataset
All the directories should be created manually before running the codes as mentioned in Directory_structure.txt file. All the folders should have both read and write access. After creating the directories with suitable permissions, competition data(only the train parquet files) must be placed in the '/data/' directory <br />
•	Image Size: 137x236 ( No preprocessing )

## Augmentation
•	CutMix

## Model
![GitHub Logo](/images/model_diagram.png)

## Training:
•	5 fold Configuration <br />
•	Data split on the basis of grapheme root labels <br />
•	Loss: Cross Entropy Loss <br />
•	Optimizer: Over9000 <br />
•	Scheduler: Reduce On Plateau <br />
•	Gradient Accumulation <br />
•	Batch Size 100 <br />
•	Initial Learning Rate 0.03 <br />

## Inference:
•	Best Average recall checkpoints were used <br />
•	Simple Average of the outputs from 5 folds <br />
•	Inference kernel: https://www.kaggle.com/mohammadzunaed/efficientnet-b5-inference-kernel-pytorch?scriptVersionId=32245517 <br />

## Things that did not work for me:
•	Preprocessing <br />
•	GridMask, Cutout, AugMix <br />
•	Label Smoothing Criterions <br />
•	Single head instead of three heads <br />
•	Activation functions and Convolutional layers in the heads <br />
