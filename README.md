# Speech Emotion Recognition using CNN
*This was our 3rd year Mini Project. We've built a Speech to Emotion Classifier using a Convolutional Neural Network.*

Emotion recognition is the part of speech recognition that is rapidly becoming popular. Although there are methods to recognize emotion using machine learning techniques, this project attempts to use deep learning to recognize the emotions from data. 

The dataset chosen was [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)(Ryerson Audio-Visual Database of Emotional Speech and Song)

Feature extraction and data augmentation was done using LIBROSA. It is a Python library for analyzing audio and music.

## CNN model ##
Our model starts with a series of convolution and max-pooling layers with 256 units each.
We then push our data through a 128 unit convolution and max-pooling layer with a dropout of 0.2.
We repeat the same with a layer of 64 units.
Finally, we flatten our input and push the data through two fully connected layers: The hidden layer with 32 units and the Output layer with 8 units, with activation functions relu and softmax respectively.

## Mini Demo of Final project ##

https://user-images.githubusercontent.com/54148193/182010936-8522395b-03a7-45ec-98ef-8c1bf7b14b9d.mp4

