import argparse
import dotenv
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
import json
from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor,
                          Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer)
from audiomentations import Compose, PitchShift, AddBackgroundNoise, TimeStretch
from typing import Union, Optional, Any, Dict, List
from dataclasses import dataclass
import torch
import numpy as np
import jiwer
import re
import os
import wandb

# local imports
# import prepare_youtube_dataset

dotenv.load_dotenv()


def load_data_from_hf(dataset_name: str) -> Dataset:
    """Load the dataset from huggingface"""
    username = "ctaguchi"
    dataset = load_dataset(f"{username}/{dataset_name}")
    return dataset["train"]


def load_data_locally(dataset_path: str) -> Dataset:
    """Load the dataset locally."""
    dataset = load_from_disk(dataset_path)
    return dataset["train"]


def generate_data() -> Dataset:
    ...


def remove_tags(batch: Dict[str, str | dict]) -> dict:
    """Count the total number of characters in an eaf annotation.
    Ignore whitespaces.
    """
    batch["text"] = re.sub(r"</?(ja|dis|unsure|song|name)>", "", batch["text"])
    return batch


def make_vocab(dataset: Dataset) -> set:
    """Get all the character types that appear in the dataset."""
    vocab = set()
    for transcription in dataset["text"]:
        vocab.update(transcription)
    
    return vocab


def prepare_vocab(dataset: Dataset,
                  phonemic_vocab: bool) -> str:
    """Prepare vocab for training."""
    vocab_file = "vocab.json"
    vocab = make_vocab(dataset)
    
    if phonemic_vocab:
        # use a predefined digraph kana set
        with open("kana_vocab.txt", "r") as f:
            phonemes = f.read().splitlines()
        vocab.update(set(phonemes))

    vocab_dict = {v: k for k, v in enumerate(vocab)}

    # Replace " " (whitespace) with a pipe "|"
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    # Add special characters
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_file, "w") as f:
        json.dump(vocab_dict, f)
    
    return vocab_file


def prepare_dataset(batch: dict,
                    augmentor=None) -> dict:
    """Prepare the dataset for the training.
    Add `input_values` and `labels` to the dataset.
    """
    audio = batch["audio"]
    if augmentor is not None: # data augmentation
        audio["array"] = augmentor(samples=audio["array"],
                                   sample_rate=audio["sampling_rate"])

    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0] # batched output is un-batched

    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = batch["text"]

    return batch


def compute_metrics(pred) -> Dict[str, float]:
        """Compute the evaluation score (CER)."""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids,
                                           group_tokens=False)

        cer = jiwer.cer(
            reference=label_str,
            hypothesis=pred_str
        )
        wer = jiwer.wer(
            reference=label_str,
            hypothesis=pred_str
        )
        
        # log full results
        table = wandb.Table(columns=["prediction", "reference"])
        for p, r in list(zip(pred_str, label_str)):
            table.add_data(p, r)
        wandb.log({"val/examples": table})

        return {"cer": cer, "wer": wer}


Feature = Dict[str, Union[List[int], torch.Tensor]]


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set, it will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self,
                 features: List[Feature]) -> Dict[str, torch.Tensor]:
        """Split inputs and labels since they have to be of different lengths
        and need different padding methods
        """
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]

        label_texts = [feature["labels"] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor(
            text=label_texts,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        batch["labels"] = labels

        return batch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Dataset group
    parser.add_argument(
        "--dataset",
        type=str,
        default="ikema_asr",
        help="The name of the dataset to use",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        help="The evaluation (validation) dataset."
    )
    parser.add_argument(
        "--generate_dataset",
        action="store_true",
        help="Generate a dataset upon training instead of loading from HF"
    )
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        help="Load a dataset locally."
    )
    parser.add_argument(
        "--use_dict_dataset",
        action="store_true",
        help="Whether to use a dictionary dataset."
    )
    parser.add_argument(
        "--dict_dataset_path",
        type=str,
        default="ikema_dict_asr",
        help="Path to the local dictionary dataset."
    )

    # Data augmentation group
    parser.add_argument(
        "--pitch_shift",
        action="store_true",
        help="Whether to apply pitch shift.",
    )
    parser.add_argument(
        "--min_semitones",
        type=float,
        default=0.0,
        help="Minimum semitones for pitch shift (lowering).",
    )
    parser.add_argument(
        "--max_semitones",
        type=float,
        default=10.0,
        help="Maximum semitones for pitch shift (raising).",
    )
    parser.add_argument(
        "--time_stretch",
        action="store_true",
        help="Whether to apply time stretch.",
    )
    parser.add_argument(
        "--min_rate",
        type=float,
        default=0.8,
        help="Minimum rate for time stretch.",
    )
    parser.add_argument(
        "--max_rate",
        type=float,
        default=1.25,
        help="Maximum rate for time stretch.",
    )
    parser.add_argument(
        "--add_background_noise",
        action="store_true",
        help="Whether to apply background noise.",
    )
    
    # Training group
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/wav2vec2-xls-r-300m",
        help="The name of the model to use",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="Number of epochs to run.",
    )
    parser.add_argument(
        "--freeze_feature_encoder",
        action="store_true",
        help="Whether to freeze the feature encoder.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--phonemic_vocab",
        action="store_true",
        help="Whether to use a phonemic vocabulary.",
    )
    
    # Misc group
    parser.add_argument(
        "--repo_name",
        type=str,
        default="wav2vec2-xls-r-300m-ikema",
        help="The name of the repository to use",
    )
    parser.add_argument(
        "--script",
        type=str,
        choices=["kana", "romaji", "phoneme"],
        default="kana",
        help="The writing system of the transcription."
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of CPUs."
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ikema-asr",
        help="WandB run name."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, the fine-tuned model will be pushed to Hugging Face Hub."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.generate_dataset:
        raise NotImplementedError

    if args.load_from_disk:
        dataset = load_data_locally(args.dataset)
    else:
        dataset = load_data_from_hf(args.dataset)

    if args.eval_dataset:
        eval_dataset = load_data_from_hf(args.eval_dataset)
        print("Loaded eval dataset:", args.dataset)
    else:
        additional_data = load_data_from_hf("ikema_youtube_asr_test") # add the youtube test set for more data
        lecture_data = load_data_from_hf("ikema_youtube_asr_hougenkougi")
        dataset = concatenate_datasets([dataset, additional_data, lecture_data])
        if args.use_dict_dataset:
            dict_dataset = load_data_from_hf(args.dict_dataset_path)
            # Rename the dictionary column names
            # originally, it has: "word_id", "word", "audio", "part_of_speech", "description", "example_sentence".
            # we only need "word" and "audio"
            dict_dataset = dict_dataset.rename_column("word", "transcription")
            dict_dataset = dict_dataset.remove_columns(["word_id", "part_of_speech", "description", "example_sentence"])
            dataset = concatenate_datasets([dataset, dict_dataset])
            print("Using dictionary dataset:", args.dict_dataset_path)
            
        train_devtest = dataset.train_test_split(test_size=0.2, seed=42)
        test_valid = train_devtest["test"].train_test_split(test_size=0.5, seed=42)
        
        dataset_dict = DatasetDict({
            "train": train_devtest["train"],
            "dev": test_valid["train"],
            "test": test_valid["test"]
            })
        
        # match the variables
        dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["dev"]

    print("Loaded dataset:", args.dataset)
    print("Dataset size:", sum([len(split) for split in dataset_dict.values()]))
    print("Training data size:", len(dataset))
    print("Dev data size:", len(eval_dataset))
    
    if args.script == "kana":
        dataset = dataset.rename_column("transcription", "text")
        eval_dataset = eval_dataset.rename_column("transcription", "text")
    elif args.script == "romaji":
        dataset = dataset.rename_column("romaji", "text")
        eval_dataset = eval_dataset.rename_column("romaji", "text")
    elif args.script == "phoneme":
        dataset = dataset.rename_column("phoneme", "text")
        eval_dataset = eval_dataset.rename_column("phoneme", "text")
    else:
        raise ValueError("Invalid script. Choose from 'kana', 'romaji', or 'phoneme'.")

    # wandb login
    try:
        wandb_api_key = os.environ["WANDB_API_KEY"]
    except KeyError as e:
        print("WandB API key not found in the environment.")
        print(e)
    
    wandb.login(key=wandb_api_key)

    dataset = dataset.map(remove_tags,
                          num_proc=args.num_proc)
    eval_dataset = eval_dataset.map(remove_tags,
                                    num_proc=args.num_proc)

    vocab_file = prepare_vocab(dataset,
                               phonemic_vocab=args.phonemic_vocab)

    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    if (args.pitch_shift or args.add_background_noise or args.time_stretch):
        augment_methods = [
            AddBackgroundNoise(sounds_path="../data/background_noise") if args.add_background_noise else None,
            PitchShift(min_semitones=args.min_semitones,
                       max_semitones=args.max_semitones,
                       p=args.pitch_shift) if args.pitch_shift else None,
            TimeStretch(min_rate=args.min_rate,
                        max_rate=args.max_rate,
                        p=args.time_stretch) if args.time_stretch else None
        ]
        augmentor = Compose([method for method in augment_methods if method is not None])

    else:
        augmentor = None
    dataset = dataset.map(prepare_dataset,
                          fn_kwargs={"augmentor": augmentor},
                          remove_columns=dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_dataset,
                                    remove_columns=eval_dataset.column_names)
    
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder() # prevents overfitting, faster training

    training_args = TrainingArguments(
        output_dir=args.repo_name,
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=args.epoch,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        hub_token=os.environ["HF_TOKEN"],
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    # dataset_dict = dataset.train_test_split(test_size=0.1)
    # train = dataset_dict["train"]
    # valid = dataset_dict["test"]

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()

    if args.push_to_hub:
        model.push_to_hub(args.repo_name,
                          use_auth_token=os.environ["HF_TOKEN"])
        tokenizer.push_to_hub(args.repo_name,
                              use_auth_token=os.environ["HF_TOKEN"])
