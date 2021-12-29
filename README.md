# MAPLE
The Official Code for Offline Model-based Adaptable Policy Learning


# [optional] Download Resources

For better reproducibility, we uploaded a backup of dataset which is used in our experiment, since we found that the content of dataset in D4RL and NeoRL might be changed.   
- D4RL: https://drive.google.com/drive/folders/1kgNg6xLHRTyb_tzDQULezB9XYGNuakCM?usp=sharing
- NeoRL: https://drive.google.com/drive/folders/1gZdVQTY_7FLCFGqszHF9sfKcXT8epoze?usp=sharing

After downloaded, you can push the data of D4RL to ~/.d4rl/datasets and NeoRL to {your path to MAPLE}/neorl_data/

We have also upload the dynamics models for MAPLE training, which can be found in:

# Installation

We use RLAssistant to manage our experiments. You can download and install it via:
```
git clone https://github.com/xionghuichen/RLAssistant.git
cd RLAssistant
pip install -e .
```
Then you can install MAPLE via:
```
git clone https://github.com/xionghuichen/MAPLE.git
cd RLAssistant
pip install -e .
```

# Quick Start

You can train your MAPLE policy directly like this:
```
cd run_scripts
# train the MAPLE policy for the hopper_low task in neorl
python main.py --config examples.config.neorl.hopper_low
or 

# train the MAPLE policy for walker2d_medium_expert task in d4rl
python main.py --config examples.config.d4rl.walker2d_medium_expert 

# train the MAPLE policy for walker2d_medium_expert task in d4rl with 200 dynamics models
python main.py --config examples.config.d4rl.walker2d_medium_expert --maple_200


# train the MAPLE policy for walker2d_medium_expert task in d4rl with your custom configs
python main.py --config examples.config.d4rl.walker2d_medium_expert --custom_config --penalty_coeff 1.0
```

The training logs can be found in {your MAPLE path}/log. You can use tensorbard to check and also use the tools in RLA to visualize.
There are also some scrips in ``./rla_scrips`` to manage the experimental logs. 

# Refernece

 