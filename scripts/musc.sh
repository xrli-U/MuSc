# MVTec AD
python examples/musc_main.py --device 0 \
--data_path ./data/mvtec_anomaly_detection/ --dataset_name mvtec_ad --class_name ALL \
--backbone_name ViT-L-14-336 --pretrained openai --feature_layers 5 11 17 23 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True

# VisA
python examples/musc_main.py --device 0 \
--data_path ./data/visa/ --dataset_name visa --class_name ALL \
--backbone_name ViT-L-14-336 --pretrained openai --feature_layers 5 11 17 23 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True

# BTAD
python examples/musc_main.py --device 0 \
--data_path ./data/btad/ --dataset_name btad --class_name ALL \
--backbone_name ViT-L-14-336 --pretrained openai --feature_layers 5 11 17 23 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True

