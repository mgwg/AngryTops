Install
=======

Setup the environment

```
virtualenv env-ml
source  env-ml/bin/activate
pip install keras
pip install tensorflow-gpu
pip install sklearn
pip install pandas
```

Install ROOT from src
https://root.cern.ch/downloading-root

Let's assume ROOT is installed under ```$HOME/local/root```

Clone repository

```
git clone https://gitlab.cern.ch/disipio/AngryTops
```

Execute
=======

```
module load cuda/9.0.176
module load cudnn
source $HOME/env-ml/bin/activate
source $HOME/local/root/bin/thisroot.sh 
```

Format of csv File
=====
- Each row represents a different event. The axis for each entry are as follows:
Number of Jets | Number of B Jets Tagged | Lepton P_x | Lepton P_y | Lepton P_z | Lepton Energy | Missing Transverse Energy | 
Missing Phi | Jet1 P_x | ... | Jet1 P_z | !!Jet1 E!! | !!Jet1 M!! | Jet1 BTag | ... | Jet5 P_z | !!Jet5 E!! | !!Jet5 M!! | Jet5 BTag | W_had P_x | ... | W_had P_z | W_had E | W_had M | | W_lep P_x | ... | W_lep P_x | ... | W_lep P_z | W_lep E | W_lep M | ... | <b_had> | <b_lep> | <t_had> | <t_lep> | 

