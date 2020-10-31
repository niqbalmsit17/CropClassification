# CropClassification

### Data set

- Images data of crops resized and converted to gray scale images is store in Crops folder

### GLCM based textures

- GLCM based harlick features generated from gray scale images are stored in Textures folder

## Generation of texural features

- glcm_texture_generation.py file contains code to generate glcm features and store in Textures folder

## Classification based on machine learning and neural network algorithms

- Machine_Learning_Algorithms_Gray_scale_vs_GLCM.ipynb
  - Contains Naive bayes, SVM and Random Forest classifier applied on gray scale images and GLCM based images
- Neural_Network_Classification_Gray_scale_vs_GLCM.ipynb
  - Contains Neural Network classifier applied on gray scale images and GLCM based images
