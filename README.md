## Simplified OCR
The SimplifiedOCR Project aims to simplify the development and training of optical character recognition (OCR) models by addressing the common challenges associated with dataset preparation. Many developers struggle with sourcing diverse and high-quality training data, which can hinder the performance of OCR systems. SimplifiedOCR offers a user-friendly framework that allows users to easily generate synthetic datasets tailored to their specific needs. By providing customizable options for text content, fonts, and image characteristics, the project empowers users to create rich, varied datasets quickly. This not only accelerates the model training process but also enhances the accuracy and robustness of OCR applications across different domains.

<p align="center">
  <img src="https://github.com/m-usman98/SimplifiedOCR/blob/main/script/Sample.jpg" width="800"/>
</p>

### Libraries Requirement
Install the following libraries to train your model and create a synthetic dataset.
```angular2html
torch 2.2.2
pandas 2.2.1
nltk 3.8.1
pillow 9.5.0
trdg 1.8.0
```

### Disclaimer
The project has been tested exclusively with the English language and may not support other languages.

### Acknowledge
This repository is a modified version of [EasyOCR](https://github.com/JaidedAI/EasyOCR/blob/master/README.md). Please visit the [EasyOCR Repo](https://github.com/JaidedAI/EasyOCR/tree/master) repository to customize ```model_arch.py``` according to your preferences.
