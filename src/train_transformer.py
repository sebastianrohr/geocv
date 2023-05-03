import torch
from torch import cuda

from argparse import ArgumentParser

import wandb
wandb.login()

from transformers import TrainingArguments, Trainer, ViTForImageClassification, ViTFeatureExtractor
from datasets import load_dataset

import numpy as np
import evaluate

from load_data import train_test_set_loader

panos = False
model_name_or_path = 'google/vit-base-patch16-224-in21k'

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['pixel_values']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def arg_parser():
    parser = ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--output_dir', type=str, default="vit-base-cities")
    parser.add_argument('--data_load', action='store_true')
    parser.add_argument('--no-data_load', dest='data_load', action='store_false')
    parser.set_defaults(data_load=True)
    args = parser.parse_args()
    return args


def data_collection(data_dir='data'):
    
    # Location of data
    datadir = data_dir

    # Datasets from each folder
    data = load_dataset("imagefolder", data_dir=datadir)
    datasets_processed = data.rename_column('image', 'pixel_values')

    prepared_data = datasets_processed.with_transform(transform)

    return prepared_data

def hyperparameter_sweep(output_dir, data_dir):

    data_dir = data_dir

    output_dir = output_dir
    data_load = data_load

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')

    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    prepared_data = data_collection(data_dir)
    labels = prepared_data["train"].features['label'].names

    model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=len(labels),
                id2label={str(i): c for i, c in enumerate(labels)},
                label2id={c: str(i) for i, c in enumerate(labels)}
            )

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_accuracy", "goal": "maximize"},
        "parameters": {"lr": {"min": 1e-6, "max": 1e-4}, "epochs": {"min": 3, "max": 10}, "batch_size": {"values": [16, 32, 48]}},
        "early_terminate": {"type": "hyperband", "s": 2, "eta": 3, "max_iter": 10}
    }

    sweep_id = wandb.sweep(sweep_config, project="geocv")

    def run(config=None):
        with wandb.init(config=config):
            config = wandb.config
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy = "steps",
                save_steps=100,
                eval_steps = 100,
                save_total_limit = 3,
                learning_rate = config.lr,
                per_device_train_batch_size = config.batch_size,
                num_train_epochs = config.epochs,
                fp16 = True,
                push_to_hub=False,
                report_to='wandb',
                run_name=f"lr_{config.lr}_batch_{config.batch_size}_epochs_{config.epochs}_output_dir_{output_dir}",
                load_best_model_at_end=True,
                metric_for_best_model = "eval_accuracy",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=prepared_data["train"],
                eval_dataset=prepared_data["validation"],
                tokenizer=feature_extractor,
            )

            trainer.train()

            wandb.log({"eval_accuracy": trainer.evaluate(prepared_data["validation"])})

    wandb.agent(sweep_id=sweep_id, function=run)


def train_model(output_dir, data_dir, config):

    learning_rate = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    # Location of data and output for model
    data_dir = data_dir
    output_dir = output_dir

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')

    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    # creting a dataset
    prepared_data = data_collection(data_dir)
    labels = prepared_data["train"].features['label'].names

    # loading pretrained model and defining labels
    model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=len(labels),
                id2label={str(i): c for i, c in enumerate(labels)},
                label2id={c: str(i) for i, c in enumerate(labels)}
            )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy = "steps",
        save_steps=100,
        eval_steps = 100,
        save_total_limit = 3,
        learning_rate = learning_rate,
        per_device_train_batch_size = batch_size,
        num_train_epochs = epochs,
        fp16 = True,
        push_to_hub=False,
        report_to='wandb',
        run_name=f"lr_{learning_rate}_batch_{batch_size}_epochs_{epochs}_output_dir_{output_dir}",
        load_best_model_at_end=True,
        metric_for_best_model = "eval_accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_data["train"],
        eval_dataset=prepared_data["validation"],
        tokenizer=feature_extractor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_data["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == '__main__':

    
    # read args
    args = arg_parser()

    # batch_size = args.batch_size
    # epochs = args.epochs
    # lr = args.lr

    data_dir = args.data_dir

    output_dir = args.output_dir
    data_load = args.data_load
    if data_load:
        train_test_set_loader(test_size=0.25, val_size=0.25, panos=panos)

    hyperparameter_sweep(output_dir, data_dir)
