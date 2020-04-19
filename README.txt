*** All the directories should be created manually before running the codes as mentioned in Directory_structure.txt file ***
*** All the folders should have both read and write access ***

After creating the directories with suitable permissions, competition data(only the train parquet files) must be placed in the
'/home/bengalidata/data/' directory

After that, by running the:

sudo python3 /home/bengalidata/bengali_train_fold1.py
sudo python3 /home/bengalidata/bengali_train_fold2.py
sudo python3 /home/bengalidata/bengali_train_fold3.py
sudo python3 /home/bengalidata/bengali_train_fold4.py
sudo python3 /home/bengalidata/bengali_train_fold5.py

codes will generate the models under the appropriate directories. Best Average recall checkpoints should be used for generating test predictions.
[Note: sudo command depends on the os/conda environment requirements, sudo will do for GCP server, 
whereas sudo should not be used for AWS ubuntu server with conda environment] 

Hardware Used:
GPU: 1xV100 NVIDIA GPU
CPU: 8 VCPU
OS: Debian[GCP], Ubuntu 18.04[AWS]


