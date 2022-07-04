# DCCDAE
## Required Environment
tensorflow == 2.4.0

python == 3.7

numpy == 1.19.5

sklearn == 0.0
## Dataset
The dataset can be downloaded from here 

[GPDS_signet_f](https://drive.google.com/open?id=1x-OnstvAMP7rw01T7Z_C7XP_i7l0TPVx) 

[MCYT_signet_f](https://drive.google.com/open?id=17BtvIbOWRk4C8xzWpBcn_K16y4FF_UEI)

[CEDAR_signet_f](https://drive.google.com/open?id=1bVnnBQPBaKkJHeXG-5idp-LV7jXSfbZY)

[brazilian_signet_f](https://drive.google.com/open?id=1sNBVk77ipBUePbC72kuS9dsrEP3zhT7e)

You need download the 
## Usage
Generate two views
```
python 1.Generate the second view.py
```
Train DCCDAE and Verification
```
python 2.3.DDCCAE、WD.py
```
All parameters used in the paper have been set to the default values and can be adjusted using the following method:
```
python 2.3.DDCCAE、WD.py --<parameter name> <parameter value>
```
