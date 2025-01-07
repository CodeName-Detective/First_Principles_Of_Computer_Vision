# Face
**CSE 573 Homework 4.**
#### <font color=red>You can only use opencv 4.5.4 for this homework.</font>


**task 1 validation set**
```bash
# Face detection on validation data
python task1.py --input_path data/validation_folder/images --output ./result_task1_val.json

# Validation
python ComputeFBeta/ComputeFBeta.py --preds result_task1_val.json --groundtruth data/validation_folder/ground-truth.json
```

**task 1 test set running**

```bash
# Face detection on test data
python task1.py --input_path data/test_folder/images --output ./result_task1.json
```

**task 2 running**
```bash
python task2.py --input_path data/images_panaroma --output_overlap ./task2_overlap.txt --output_panaroma ./task2_result.png
```

**Pack your submission**
Note that when packing your submission, the script would run your code before packing.
```bash
sh pack_submission.sh <YourUBITName>
```
Change **<YourUBITName>** with your UBIT name.
The resulting zip file should be named **"submission\_<YourUBITName>.zip"**, and it should contain 6 files, named **"task1.py"**, **"task2.py"**,  **"result_task1.json"**, **"result_task1_val.json"**, **"task2_overlap.txt,"**, and **"task2_result.png"**. If not, there is something wrong with your code/filename, please go back and check.

You should only submit the zip file.
