#!/bin/bash

python3 task1.py --input_path data/validation_folder/images --output ./result_task1_val.json
python3 task1.py --input_path data/test_folder/images --output ./result_task1.json
python3 task2.py --input_path data/images_panaroma --output_overlap ./task2_overlap.txt --output_panaroma ./task2_result.png

python3 utils.py --ubit $1