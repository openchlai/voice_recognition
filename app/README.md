# Environment Setup and Runtime

## 1. Create a Python Virtual Environment
We recommend `workon` command, but you can use `virtualenv` to manage your virtual environments. Below instructions can work on `MacOS` and `Windows OS`, just make adjustment where applicable

```sh
sudo apt install python3-pip python3-virtualenv
```

## Create the following `virtualenvs` dir as hidden folder in your $HOME

```sh
mkdir $HOME/.virtualenvs
pip3 install virtualenvwrapper
```

## Update your `local env` by editing `.bashrc` file. You can use `vim` or `nano`, or any CLI text editor

```sh
nano $HOME/.bashrc
```

## Paste the following at the bottom of thr `.bashrc` file

```sh
#Virtualenvwrapper settings:
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_VIRTUALENV=/home/bitzml/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh
```

## 2. Install required libraries
We will assume you are using `mkvirtualenv` command to manage your virtual environments. Once you ativate your virtual environment, run the following command `pip install -r requirements.txt`

## 3. Sort Testing Media
The directory `media` will hold all audio for the exercise. Ideally got to YouTube and download some clips (audio-only) to for testing and drop them into `media/demo` sub-directory. Create the `demo` sub-directory.

When done with above preparations, run the following command: `python sortmedia.py`

A directory named `media` will be created. For testing, the `demo` directory will be the `source` and a new directory named `media/volume` will be created as `dest` for **destination**. You can change the `source` and `dest` directory from the file.

## 4. Reference Transcription Text
Every audio in the `source` directory needs to have a corresponding `reference` transcription which will be usd to compare with actual transcription. Save this reference on a text file named after the audio file name. For example, a file named `my classic audio.mp3` should have a corresponding `reference` text file save as `my classic audio.txt` 

## 5. Master Audio Data
After completing sorting above media, the data for processed and sorted audio files will be stored in a `master.json` file found within the base of `media` directory. Below is a quick guide on the `key-values` of `master.json`

1. `parent` is the original filename as **SHA256** encrypted
2. `models` consists of nested dictionary of actual runtime transcription per model used.
3. `reference` refers to the human-made transcription of test audios

This `dict` will form the foundation of data captured to track progress across the test exercise, with more nested `key-value` pairs added.

## 6. Visualization of Progress
A separate `python-streamlit` web app will be made available for this purpose

