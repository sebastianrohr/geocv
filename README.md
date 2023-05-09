# 🌎 GeoCV: A Geoguessr Bot in Python
This code uses a ViT (Vision Transformer) for image classification. It loads the [ViT-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) pre-trained model from the Hugging Face 🤗 Transformers library and fine-tunes it on a custom dataset.

## 🤖 For ML mini project
Relevant files are 
1. `train_transformer.py` where all the actual training happens and the hyperparameter search is defined
2. `geoguessr_guessr_bot.ipynb` for an overview, we will also use that notebook for our presentation.

## 💻 Installation
Create a new virtual environment and install the required dependencies:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## 🔍 Code
There are two modes of operation:

1. **Hyperparameter sweep**: This mode will run a hyperparameter sweep using WandB. It will search for the best combination of hyperparameters among the ones specified in hyperparameter_sweep(). To run this mode, call the function hyperparameter_sweep(output_dir, data_dir) with the desired output and data directories as arguments. Or see below for a command line interface.

2. **Train model**: This mode will train the model on the specified dataset using the hyperparameters specified in the config dictionary. To run this mode, call the function train_model(output_dir, data_dir, config) with the desired output and data directories, as well as a configuration dictionary with hyperparameters as arguments. Or see below for a command line interface.

## 📂 Data
When running the script for the first time the data_load parameter should be passed to split the images into train, validation and test set. The data directory should contain a subdirectory for each city. Each city directory should contain the images for that city.
The data directory should contain the following files and be called "output":
```
output
├── city1
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│   └── imageN.jpg
├── city2
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
│   └── imageN.jpg
├── ...
└── cityN
    ├── image1.jpg
    ├── image2.jpg
    ├── ...
    └── imageN.jpg
```


## 🎮 Usage
To perform a hyperparameter search, run the following command:

```
python3 train_transformer.py --data_dir path/to/data --output_dir path/to/output --hyperparameter_search --data_load
```

To train a model, run the following command and adjust the hyperparemeters as needed:
```
python3 train_transformer.py --data_dir path/to/data --output_dir path/to/output --batch_size 32 --learning_rate 1e-4 --num_epochs 10 --data_load
```

