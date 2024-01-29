import os
import yaml
 
def load_yaml(config_path):
    filepath = os.path.join(os.getcwd(), config_path)
    with open(filepath, 'r', encoding='UTF-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs
 
 