# image_caption_generator
Python project that recognizes the context of an image using Computer vision and Machine Learning  and speaks it out.
Developed for visually impaired people.
Deployed on AWS using load balancing.
Image features are extracted by CNN model (ResNet-50).
The LSTM model then converts the features into a caption.
Text to speech is done using the gTTS API in python.
