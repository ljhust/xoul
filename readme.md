# xoul homework
## Dir Structure
- **app**: API application dir
- **deployment**: deploy base model "sophosympatheia/Midnight-Miqu-70B-v1.0" and finetuned model
- **train**: how to fine "sophosympatheia/Midnight-Miqu-70B-v1.0" model
- **process_data**: The first step, you have to process the data first
## procedure
### install requirments
```
pip install requirments.txt
```
### data process
- make a dir named data, and sub dir raw_data, and another sub dir threads
```
mkdir data
cd data
mkdir raw_data
mkdir threads
```

- download raw data
```
cd raw_data
```
there 8 data to download 
[part1](https://files.catbox.moe/uuub3g.7z)
[part2](https://files.catbox.moe/luxmdg.7z)
[part3](https://files.catbox.moe/o4bo59.7z)
[part4](https://files.catbox.moe/wpceng.7z)
[part5](https://files.catbox.moe/exl3ia.7z)
[part6](https://files.catbox.moe/q9mys3.7z)
[part7](https://files.catbox.moe/nhhnj0.7z)
[part8](https://files.catbox.moe/gqilvh.7z)
download the data to raw_data dir
```
wget part*(url)
```
change the process_data's data.yaml file, modify the dir_path to your raw_data path. Run the process.py and processed data will be dump into threads dir

```
cd process_data
python process.py
```
### train
- Enviroment

8*A100
"python2.4 cuda12.4" image

- login wandb
In command run
```
wandb login
```
to login wandb for logging your training data and metrics

using accelerate to run the "miqu_lora_finetune.py" and you can modify config.yaml for trainig in train dir

```
accelerate launch miqu_lora_finetune.py
```
### deloy model and lora
After training, download the lora model to a specific path and record it.
- enviroment

2*A100 to run base model and lora both
"python2.4 cuda12.4" image

- modify config

modify the model and lora path in model.yaml file

and run below command to deploy

```
serve run llm:build_app tensor-parallel-size=2 accelerator="GPU" 
```

### app

run below command to run app in the root dir

```
uvicorn app.main:app --reload
```


## Instruction

In this section I will explain some thoughts and details in this project

### dataset choosing

I choose the "Roleplayer Guild" dataset cause quality level is **Excellent**, and the dataset description below. Also, compraring to other datasets, this dataset is intact and good.

> This dataset is different compared to the others in that it includes within the same .csv files in-character (IC, i.e. actual roleplay), out-of-character (OOC) and Character Sheet messages for a total of about 3 million messages. As OOC and Sheets share the same base url/name with the IC threads, they can be reliably associated with them, if needed. Thread tags and an additional field identifying if the messages are part of IC, OOC or sheets are included. Possibly one of the best all-around RP datasets. Special usage notes: 1: @-mentions in the IC threads could be removed. 2: A markdown file with an extended explanation of thread tags is provided.

### data processing
the final data will be processed as threads that each thead represents a whole story, a intact story.

- keep only IC column for in-character reason 
- Remove duplicates
- Handle malformed and misformatted entries

### training

- Take more than 300K threads of data to train, not all of the data. The whole dataset is much more bigger.
- using accelerate to distributed training by 8*A100
- Leveraging wandb to track the training process
- config.yaml to make it configrurable

### deployment

- Using Ray to manage the cluster that make it scalable
- Ray provide dashboard to check the cluster's situation
- Using vLLM to inference the model
- One base model and lora based on that base model, not need to init another base model to load lora. Saveing a lot of VRAM, this solution also can spreed to multi-lora situations as well

### app

- Input for both base model and lora and compare
- Vote feature choosing the model they like
- statistics feature evaluating preferences
- FastAPI based
- sqlite3 for simplicity and demonstraction can be trainsition to another database easily
