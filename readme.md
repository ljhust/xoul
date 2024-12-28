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
