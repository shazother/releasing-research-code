## 📋 README.md template

This code is for replicating experiments presented in the paper 'Smart Systematic Investment Agent: an ensemble of deep learning and evolutionary strategies' and submmited to Nuerips2021. 
This page contains step by step instructions on replicating the experiments presented in the paper and instructions on creating new ones to stress test the model.
The code is written purely in python and all dependencies are included in requirements.txt or as an executable script.

#### 1. Dependencies

The code is dependent on packages that can be installed by running the requirements file.
To install requirements:

```setup
pip install -r requirements.txt
```

#### 2. Creating environment, espisoded and actions 

This section covers section 2.1 - 2.4 of our paper. It generates episodic samples, does feature engineering, scaling and finds the optimal vector of action using genetic algorithm that is subsequently used as a training dataset for the nueral network.

```Generate solved episodes
python Neurips_GA_Environment_Data_Creation.py
```

#### 3. Training the neural network model

The next step is tp train the agent to learn the policy behind optimal action using a simple neural network and to save the model to disk.

```Train model and save to disk
python nuerips_nn_training_upto_2020_data.py
```

#### 4. Pre-trained models

Training a model from scratch can be time-consuming and expensive. One way to increase trust in your results is to provide a pre-trained model that the community can evaluate to obtain the end results. This means users can see the results are credible without having to train afresh.

Another common use case is fine-tuning for downstream task, where it's useful to release a pretrained model so others can build on it for application to their own datasets.

Lastly, some users might want to try out your model to see if it works on some example data. Providing pre-trained models allows your users to play around with your work and aids understanding of the paper's achievements.

#### 5. README file includes table of results accompanied by precise command to run to produce those results

Adding a table of results into README.md lets your users quickly understand what to expect from the repository (see the [README.md template](templates/README.md) for an example). Instructions on how to reproduce those results (with links to any relevant scripts, pretrained models etc) can provide another entry point for the user and directly facilitate reproducibility. In some cases, the main result of a paper is a Figure, but that might be more difficult for users to understand without reading the paper. 

You can further help the user understand and contextualize your results by linking back to the full leaderboard that has up-to-date results from other papers. There are [multiple leaderboard services](#results-leaderboards) where this information is stored.  

## 🎉 Additional awesome resources for releasing research code

### Hosting pretrained models files

1. [Zenodo](https://zenodo.org) - versioning, 50GB, free bandwidth, DOI, provides long-term preservation
2. [GitHub Releases](https://help.github.com/en/github/administering-a-repository/managing-releases-in-a-repository) - versioning, 2GB file limit, free bandwidth
3. [OneDrive](https://www.onedrive.com/) - versioning, 2GB (free)/ 1TB (with Office 365), free bandwidth
4. [Google Drive](https://drive.google.com) - versioning, 15GB, free bandwidth
5. [Dropbox](https://dropbox.com) - versioning, 2GB (paid unlimited), free bandwidth
6. [AWS S3](https://aws.amazon.com/s3/) - versioning, paid only, paid bandwidth
7. [huggingface_hub](https://github.com/huggingface/huggingface_hub) - versioning, no size limitations, free bandwidth
8. [DAGsHub](https://dagshub.com/) - versioning, no size limitations, free bandwith
9. [CodaLab Worksheets](https://worksheets.codalab.org/) - 10GB, free bandwith
 
### Managing model files

1. [RClone](https://rclone.org/) - provides unified access to many different cloud storage providers

### Standardized model interfaces

1. [PyTorch Hub](https://pytorch.org/hub/)
2. [Tensorflow Hub](https://www.tensorflow.org/hub)
3. [Hugging Face NLP models](https://huggingface.co/models)
4. [ONNX](https://onnx.ai/)

### Results leaderboards

1. [Papers with Code leaderboards](https://paperswithcode.com/sota) - with 4000+ leaderboards
2. [CodaLab Competitions](https://competitions.codalab.org/) - with 450+ leaderboards
3. [EvalAI](https://eval.ai/) - with 100+ leaderboards
4. [NLP Progress](https://nlpprogress.com/) - with 90+ leaderboards
5. [Collective Knowledge](https://cKnowledge.io/reproduced-results) - with 40+ leaderboards
6. [Weights & Biases - Benchmarks](https://www.wandb.com/benchmarks) - with 9+ leaderboards

### Making project pages

1. [GitHub pages](https://pages.github.com/)
2. [Fastpages](https://github.com/fastai/fastpages)

### Making demos, tutorials, executable papers

1. [Google Colab](https://colab.research.google.com/)
2. [Binder](https://mybinder.org/)
3. [Streamlit](https://github.com/streamlit/streamlit)
4. [CodaLab Worksheets](https://worksheets.codalab.org/)

## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at hello@paperswithcode.com or open an issue on this GitHub repository. 

All contributions welcome! All content in this repository is licensed under the MIT license.
