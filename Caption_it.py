#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences

# In[2]:


model = load_model("storage/model_19.h5")
#model._make_predict_function()
# after removing this code is working fine on my pc and heroku web

# In[3]:


model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))



# In[4]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
#model_resnet._make_predict_function()
# after removing this code is working fine on my pc and heroku web


# In[5]:


def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# In[16]:


def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
    return feature_vector


# In[19]:


with open("./storage/word_to_idx.pkl", 'rb') as w2i:
    word_to_idx = pickle.load(w2i)
    
with open("./storage/idx_to_word.pkl", 'rb') as i2w:
    idx_to_word = pickle.load(i2w)


# In[20]:


# in my case max_len is 35 captured in image caption bot programe
def predict_caption(photo):
    in_text = "startseq"
    max_len = 35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[21]:


def caption_this_image(image):
    enc = encode_image(image)
    caption = predict_caption(enc)
    
    return caption


# In[ ]:




