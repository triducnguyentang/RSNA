# Kaggle competition: RSNA Intracranial Hemorrhage Detection

This is my single model: InceptionV3 with Deep Supervision, a part of our 5th place solution. 
(For more detail please refer this link: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117232#latest-674608)

To train the model:
$ python main_tf2.py

To generate OOF data:
$ python generate_oof.py

To make prediction:
$ python predict_tf2.py
