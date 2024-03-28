# CRISPert
This repository contains the code for the Crispr off-target detection model CRISPert
# Requirements
  - python==3.6
  - torch==1.9.0+cu111
  - torchvision==0.10.0+cu111
  - Pandas
    

# Setup
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# Pretrained model
The pretrained model can be downloaded from https://drive.google.com/drive/folders/1d7fzdoi-GnZAyJUiCANqDDnrUYpEQZl8?usp=sharing
To use it, extract the folder "pretrained_model" to the CRISPert folder.

# File Description
- finetune_model.py contains the code for loading data, training, evaluation and prediction. It can be run to test the code on a simple 80-20 train-test split of the dataset.
- process_data.py contains functions that convert the DeepCRISPR and Caskas datasets to model input data found in /data. It should be run to generate this input data for both leave-one-sgRNA-out testing scenarios.
- leave_one_sgRNA_out_testing.py is used for test scenario 1 as described in the paper. It uses the base CRISPert model without CasKas features and the DeepCRISPR dataset.
- leave_one_sgRNA_out_caskas_testing.py is used for test scenario 2 as described in the paper. It uses the CRISPert model with CasKas features and the CasKas dataset.
  
