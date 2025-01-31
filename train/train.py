#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (
    unicode_literals,
    print_function
    )

# import os
import torch
import evaluate

from time import (
    strftime)
from dataclasses import (
    dataclass)
from typing import (
    Any, Dict, List, Union)
from datasets import (
    Audio,
    DatasetDict,
    load_dataset
    )
from transformers import (
    Seq2SeqTrainer,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration
    )


print(strftime('%d %b %Y %H:%M:%S'), "Initialize Training\n")
# Since Swahili is very low-resource, we'll combine the train and validation 
# splits to give approximately 8 hours of training data. We'll use the 4 
# hours of test data as our held-out test set:

common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "sw",
    split="train+validation"
    )
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "sw",
    split="test"
    )

# print(strftime('%H:%M:%S'), common_voice)

print(strftime('%H:%M:%S'), "Remove unwanted Columns")
COLS = ["accent", "age", "client_id", "down_votes"]
COLS += ["gender", "locale", "path", "segment", "up_votes"]

common_voice = common_voice.remove_columns(COLS)

print(strftime('%H:%M:%S'), common_voice['train'][0])

print("\nThe ASR Pipeline\n")
"""
ASR Pipeline
1. A feature extractor which pre-processes the raw audio-inputs (WhisperFeatureExtractor)
2. The model which performs the sequence-to-sequence mapping (WhisperTokenizer)
3. A tokenizer which post-processes the model outputs to text format
"""

print(strftime('%H:%M:%S'), "1. Load WhisperFeatureExtractor")
# 1. Pads / truncates the audio inputs to 30s: any audio inputs shorter than 30s are 
# padded to 30s with silence (zeros), and those longer that 30s are truncated to 30s
# 2. Converts the audio inputs to log-Mel spectrogram input features, a visual representation
# of the audio and the form of the input expected by the Whisper model
# Load the feature extractor from the pre-trained checkpoint
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small")

print(strftime('%H:%M:%S'), "2. Load WhisperTokenizer")
# Load the pre-trained tokenizer and use it for fine-tuning
# Inform the tokenizer to prefix the language and task tokens to the start of encoded label sequences
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Swahili",
    task="transcribe"
    )

# verify tokenizer correctly encodes
input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


print(strftime('%H:%M:%S'), "Combine Feature Extractor with Tokenizer")

# we only need to keep track of two objects during training: the processor and the model
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Swahili",
    task="transcribe"
    )

print(strftime('%H:%M:%S'), "Prepare Data\n")

# Downsample input audio from 48kHz to 16kHz
common_voice = common_voice.cast_column(
    "audio",
    Audio(sampling_rate=16000)
    )


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
        ).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


print(strftime('%H:%M:%S'), "Map data prep\n")
# Set num_proc=1, to be safe!
# We can apply the data preparation function to all of our training 
# examples using dataset's .map method:

"""
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1
    )
# """

print(strftime('%H:%M:%S'), "The Training\n")
"""
1. Load a pre-trained checkpoint: we need to load a pre-trained checkpoint 
and configure it correctly for training.
2. Define a data collator: the data collator takes our pre-processed data and 
prepares PyTorch tensors ready for the model.
3. Evaluation metrics: during evaluation, we want to evaluate the model using 
the word error rate (WER) metric. We need to define a compute_metrics function 
that handles this computation.
4. Define the training configuration: this will be used by the Trainer to 
define the training schedule.
"""

print(strftime('%H:%M:%S'), "1. Load a Pre-Trained Checkpoint\n")

# load weights from the Hugging Face Hub
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small")

# force the model to generate in Swahili with task
model.generation_config.language = "swahili"
model.generation_config.task = "transcribe"

# legacy way of setting the language and task arguments
model.generation_config.forced_decoder_ids = None

print(strftime('%H:%M:%S'), "Define a Data Collator\n")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int],
        torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths 
        # and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

print("\nInitialise the data collator defined 👆🏻️\n")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id
    )

print(strftime('%H:%M:%S'), "Evaluation Metrics: Word Error Rate (WER)\n")

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=True)
    label_str = tokenizer.batch_decode(
        label_ids,
        skip_special_tokens=True)

    wer = 100 * metric.compute(
        predictions=pred_str,
        references=label_str
        )

    return {"wer": wer}


print(strftime('%H:%M:%S'), "Define the Training Configuration")

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-sw",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,  # don't push to hub
)

# forward the training arguments to the Trainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("\nSave Processor before Training\n")
# save the processor object once before starting training.
# Since the processor is not trainable, it won't change

processor.save_pretrained(training_args.output_dir)

print(strftime('%H:%M:%S'), "The Training\n")

# We now need launch training:

# trainer.train()

print(strftime('%d %b %Y %H:%M:%S'), "Exiting Training\n")
