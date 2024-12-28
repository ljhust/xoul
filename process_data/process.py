import pandas as pd
import yaml
from typing import Dict

def load_yaml_config(yaml_path: str) -> Dict:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"can't find file: {yaml_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML wrong format: {str(e)}")
    
config = load_yaml_config("data.yaml")

file_list = [
    "exl3ia/roleplayerguild--all_char-ic-ooc_part5.csv",
    # "gqilvh/roleplayerguild--all_char-ic-ooc_part8.csv",
    # "luxmdg/roleplayerguild--all_char-ic-ooc_part2.csv",
    # "nhhnj0/roleplayerguild--all_char-ic-ooc_part7.csv",
    # "o4bo59/roleplayerguild--all_char-ic-ooc_part3.csv",
    # "q9mys3/roleplayerguild--all_char-ic-ooc_part6.csv",
    # "uuub3g/roleplayerguild--all_char-ic-ooc_part1.csv",
    # "wpceng/roleplayerguild--all_char-ic-ooc_part4.csv"
]

# modify here if root dir of data change
dir_path = config["dir_path"]


for file_name in file_list:
    file_path = f"{dir_path}{file_name}"
    df = pd.read_csv(file_path)
    df1 = df[df['thread_type']=="IC"] # only IC data remain
    for title, group in df1.groupby('thread_title'):
    # 创建一个安全的文件名（移除非法字符）
        safe_title = "".join(x for x in title if x.isalnum() or x in (' ', '-', '_'))[:100]
        if not safe_title.strip():
            continue
        # 将该组的所有messages连接并写入文件
        with open(f'./data/threads/{safe_title}.txt', 'w', encoding='utf-8') as f:
            messages = group['message'].str.replace(r'<[^>]*>', '', regex=True)  # 移除HTML标签
            messages = [str(item) for item in messages]
            # save space
            messages = messages[:3]
            f.write('\n'.join(messages))

print("data process finished!")
   