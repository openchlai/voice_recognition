# Evaluating Automated Speech Recognition (ASR) Models

*Accuracy* is a common metric for classifiers which measures the fraction of classifications that the model gets correct on a testing dataset.

## Install required libraries
`Whisper` is a free and open-source automatic speech model, it represents the state-of-the-art for build speech recognition systems.

`JiWER` is a Python library to evaluate ASR system. It supports the following measures:

WER: Word Error Rate
MER: Match Error Rate
WIL: Word Information Lost
WIP: Word Information Preserved
CER: Character Error Rate

```sh
pip install openai-whisper jiwer
```

All metrics are computed between `reference` by a human and `hypothesis` transcriptions

## Transcribing Audio

```python
import jiwer
import whisper

audio_file = "/media/dir/some-audio-file.mp3"
ref_file = "/media/dir/some-text-file.txt"

# reference  = "I am 32 *years* old and I *am* a *software* developer"
with open(ref_file, 'r') as fp:
	reference = fp.readfile()

model = whisper.load_model("turbo")
result = model.transcribe(audio_file, language="en")

hypothesis = result["text"]
print(hypothesis)

transforms = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

wer = jiwer.wer(
	reference,
    hypothesis,
    truth_transform=transforms,
    hypothesis_transform=transforms,
    )

print(f"Word Error Rate (WER) :", wer)
# hypothesis = "I am *a* 32 *year* old and I am *as* a developer"

```

## Calculating WER
reference  = "I am 32 *years* old and I *am* a *software* developer"
hypothesis = "I am *a* 32 *year* old and I am *as* a developer"

Substitutions = `1`
Insertions = `2`
Deletions = `1`
Number of word  = *12*

WER = `(1 + 2 + 1) / 12`
WER = `0.33`
