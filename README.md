## Smart Systematic Investment Agent: an ensemble of deep learning and evolutionary strategies

This code is for replicating experiments presented in the paper 'Smart Systematic Investment Agent: an ensemble of deep learning and evolutionary strategies' and submmited to Nuerips2021. 
This page contains step by step instructions on replicating the experiments presented in the paper and instructions on creating new ones to stress test the model.
The code is written purely in python and all dependencies are included in requirements.txt or as an executable script.

#### 1. Dependencies

The code is dependent on packages that can be installed by running the requirements file.
To install requirements:

```setup
pip3 install -r requirements.txt
```

#### 2. Creating environment, espisoded and actions 

This section covers section 2.1 - 2.4 of our paper. It generates episodic samples, does feature engineering, scaling and finds the optimal vector of action using genetic algorithm that is subsequently used as a training dataset for the nueral network.

The first argument takes the index fund - VTI for our case

The second argument takes the period upto which agent training data is considered - 2019-12-31

The third argument is the number of espisodes to generate - 5000

```Generate solved episodes
python3 scripts/neurips_GA_Environment_Data_Creation.py VTI 2019-12-31 5000
```

#### 3. Training the neural network model

The next step is tp train the agent to learn the policy behind optimal action using a simple neural network and to save the model to disk.
##### Please make sure that the same arguments are used as in the previous step
The first argument takes the index fund - VTI for our case (which will be the name of CSV we create in previous step)

The second argument takes the period upto which agent training data is considered - 2019-12-31 (which will be the name of CSV we create in previous step)

The third argument is the numnber of epochs for training - 150

```Train model and save to disk
python3 scripts/nuerips_nn_training_upto_2020_data.py VTI 2019-12-31 150
```
The model will be saved by the name:
'''Neurips_agent_model_CUSTOM_MODEL.h5'''

#### 4. Pre-trained models

Pre-trained model can be found under the name:

'''Neurips_agent_model_2020_before_vF.h5'''

This can only be used for validating what it has been trained on - 'VTI post 2019-12-31' 

#### 5. Validation script on out of sample

The following script needs to be run for out of sample validation. It produces daily action of the agent during the testing period and outputs a csv of action that can be analysed by the validator for performance.

'''python3 -W ignore scripts/nuerips_validation_of_model_out_of_sample.py VTI 2020-1-1 2021-1-1 PRE'''

The first argument takes the index fund - VTI for our case (which will be the name of CSV we create in previous steps)

The second argument takes the period beginning which agent validation data is considered - 2020-1-1 for us since the agent is trained till 2019-12-31

The third argument takes the period upto which agent validation data is considered - 2021-1-1 (not recommended to validate more than a year without retraining agent, although the script can be run uptill today's date)

The final argument is which model to use for validation: '''PRE''' is used for pre-trained model that will match the results provided in the paper. '''CUSTOM''' will use custom model that has been trained using the scripts provided