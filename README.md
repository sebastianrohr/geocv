# 🌎 GeoCV: A Geoguessr Bot in Python 🤖
GeoCV is a Python-based bot that plays the popular online game, Geoguessr, for you! It uses computer vision and machine learning techniques to analyze the image and guess the location.

## 🚀 Features
- Automatically plays Geoguessr using computer vision and machine learning
- Uses Google Street View API to fetch image data
- Easy-to-use command line interface
- Customizable settings for game difficulty and bot accuracy

## 💻 Installation:
1. Create a new virtual environment:
```
python3 -m venv env
source env/bin/activate
```
2. Install the required dependencies:
```
pip3 install -r requirements.txt
```

## 🎮 Usage
To use the GeoCV pipeline, run the following command:

```
python3 main.py -l paris -n 10
```
Here, the `-l` flag specifies the location and `-n` specifies number of locations you want to geolocate,

## 🤝 Contributing
Contributions to GeoCV are welcome! If you have an idea for a new feature or want to improve the code, please submit a pull request with your changes.

## 📄 License
GeoCV is licensed under the MIT License.