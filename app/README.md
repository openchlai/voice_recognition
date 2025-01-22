# Environment Setup and Runtime

## 1. Create a Python Virtual Environment

## 2. Install required libraries
We will assume you are using `mkvirtualenv` command to manage your virtual environments. Once you ativate your virtual environment, run the following command `pip install -r requirements.txt`

## 3. Sort Testing Media
The directory `media` will hold all audio for the exercise. Ideally got to YouTube and download some clips (audio-only) to for testing and drop them into `media/demo` sub-directory. Create the `demo` sub-directory.

When done with above preparations, run the following command: `python sortmedia.py`

A directory named `media` will be created. For testing, the `demo` directory will be the `source` and a new directory named `media/volume` will be created as `dest` for **destination**. You can change the `source` and `dest` directory from the file.

## 4. Master Audio Data
After completing sorting above media, the data for processed and sorted audio files will be stored in a `master.json` file found within the base of `media` directory.
