# 🌎 GeoCV: A Geoguessr Bot in Python 🤖
GeoCV is a Python-based bot that plays the popular online game, Geoguessr, for you! It uses computer vision and machine learning techniques to analyze the image and guess the location.

## 🚀 Features
- Automatically plays Geoguessr using computer vision and machine learning
- Uses Google Street View API to fetch image data
- Easy-to-use command line interface
- Customizable settings for game difficulty and bot accuracy

## 💻 Installation
Create a new virtual environment and install the required dependencies:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## 🎮 Usage
To use the scrape output data, run the following command:

```
python3 ./src/scraper.py -l paris -n 10
```
The `-l` flag specifies the location, `-n` specifies number of locations

## 📝 Todo
✅ Make Scraper

✅ Get more API requests

✅ Decide on cities - Moscow, Copenhagen, Madrid, London

✅ Run the stuff

✅ Store the data

❌ Consider preprocessing the images

❌ Build the transformer CNN

❌ Visualize the attention
