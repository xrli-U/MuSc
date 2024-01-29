import os
import numpy as np
import torch
import pickle

path = './models/image_features'
mvtec_dat_list = [f for f in os.listdir(path) if 'mvtec' in f]
visa_dat_list = [f for f in os.listdir(path) if 'visa' in f]

mvtec_dat = {}
for f_path in mvtec_dat_list:
    if 'cls' in f_path:
        continue
    with open(os.path.join(path, f_path), 'rb') as f:
        cls_tokens = pickle.load(f)
    category = f_path.replace('mvtec_ad_', '').replace('.dat', '')
    mvtec_dat[category] = cls_tokens[0]
mvtec_dat_savepath = os.path.join(path, 'mvtec_ad_cls.dat')
with open(mvtec_dat_savepath, 'wb') as f:
    pickle.dump(mvtec_dat, f)

visa_dat = {}
for f_path in visa_dat_list:
    if 'cls' in f_path:
        continue
    with open(os.path.join(path, f_path), 'rb') as f:
        cls_tokens = pickle.load(f)
    category = f_path.replace('visa_', '').replace('.dat', '')
    visa_dat[category] = cls_tokens[0]
visa_dat_savepath = os.path.join(path, 'visa_cls.dat')
with open(visa_dat_savepath, 'wb') as f:
    pickle.dump(visa_dat, f)
