# 🌎 GeoCV: A Geoguessr Bot in Python 🤖
GeoCV is a Python-based bot that plays the popular online game, Geoguessr, for you! It uses computer vision and machine learning techniques to analyze the image and guess the location.

## 🚀 Features
- Automatically plays Geoguessr using computer vision and machine learning
- Uses Google Street View API to fetch image data
- Easy-to-use command line interface
- Customizable settings for game difficulty and bot accuracy

## 📋 Requirements
- Python 3.11
- Additional Python packages: `numpy`, `opencv-python`
## ⚙️ Installation
1. Clone this repository using Git or download the ZIP file and extract it to your preferred directory.

2. Install the required packages using pip:
```pip install -r requirements.txt```

3. Create a .env file in the root directory of the project and add your Google Maps API key:
```GMAPS_API_KEY=your_api_key_here```

4. You're ready to go! Run the bot using the command:
```python geocv.py```

## 🎮 Usage
The bot will automatically play Geoguessr for you, guessing the location of each image it receives. You can customize the settings in the `config.py` file to adjust the game difficulty and bot accuracy.

To stop the bot, simply press `Ctrl + C` in the terminal.

## 🤝 Contributing
Contributions to GeoCV are welcome! If you have an idea for a new feature or want to improve the code, please submit a pull request with your changes.

## 📄 License
GeoCV is licensed under the MIT License.
