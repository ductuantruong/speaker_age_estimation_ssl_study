# A study of different SSL models on age estimation

This Repository contains the code for estimating the Age and Height of a speaker with their speech signal. This repository uses [s3prl](https://github.com/s3prl/s3prl) library to load various upstream models like wav2vec2, CPC, TERA etc. This repository uses TIMIT dataset. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

### Download the TIMIT dataset
```bash
wget https://data.deepai.org/timit.zip
unzip timit.zip -d 'path to timit data folder'
```

### Prepare the dataset for training and testing
```bash
python TIMIT/prepare_timit_data.py --path='path to timit data folder'
```

### Update Config and Logger
Update the config.json file to update the upstream model, batch_size, gpus, lr, etc.

### Training
```bash
python train_timit.py --data_path='path to final data folder' --speaker_csv_path='path to this repo/SpeakerProfiling/Dataset/data_info_height_age.csv'
```

Example:
```bash
python train_timit.py --data_path=data/wav_data/ --speaker_csv_path=Dataset/data_info_height_age.csv
```

### Testing
```bash
python test_timit.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test_timit.py --data_path=data/wav_data/ --model_checkpoint=checkpoints/epoch=1-step=245-v3.ckpt
```

### Using different self-supervised learning
To load a specific upstream SSL model, we need to pass an argument value of `--upstream_model`. Moreover, each upstream model have a different total number of encoder layers. Therefore, you can specify the encoder layer that you want to use with the parameter `--hidden_state`.

|       Model       |     Argument Value   | Pretrained Corpus | No. Encoder layers |     Feature Dim    |
|:-----------------:|:--------------------:|:-----------------:|:------------------:|:------------------:| 
| PASE+             | pase_plus            | LS 50 hr          |         8          |         256        |           
| NPC               | npc_960hr            | LS 960 hr         |         4          |         512        |
| wav2vec 2.0 Base  | wav2vec2_base_960    | LS 960 hr         |         12         |         768        |
| wav2vec 2.0 Large | wav2vec2_large_ll60k | LL 60k hr         |         24         |         1024       |
| XLSR-53           | xlsr_53              | MLS               |         24         |         1024       |
| HuBERT Base       | hubert_base          | LS 960 hr         |         12         |         768        |
| HuBERT Large      | hubert_large_ll60k   | LL 60k hr         |         24         |         1024       |
| WavLM Base        | wavlm_base           | LS 960 hr         |         12         |         768        |
| WavLM Base+       | wavlm_base_plus      | Mix 94k hr        |         12         |         768        |
| WavLM Large       | wavlm_large          | Mix 94k hr        |         24         |         1024       |
| data2vec Base     | data2vec_base_960    | LS 960 hr         |         12         |         768        |
| data2vec Large    | data2vec_large_ll60k | LL 60k hr         |         24         |         1024       |

Example:
```bash
python train_timit.py --upstream_model=npc_960hr --hidden_state=4
```
**_NOTE:_**  If you want to run with PASE+ and NPC, please checkout the `pase+` and `npc` branch and follow the README in that branch. The other SSL models can be run with `main` branch.

### Pretrained Model
We have uploaded pretrained models of our experiments. You can download the from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/ductuan001_e_ntu_edu_sg/EhgacD3UO4tDnzB-VH8T6lYBtSiuUqG2PwKPRTehA6m8lA?e=pv8nYz).

Download it and put it into the model_checkpoint folder.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen

