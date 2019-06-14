from generators import ImageGenerator
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, Flatten, Dense, Reshape, Input, Activation, Dropout, GaussianDropout
import string
from keras.models import load_model
from keras.preprocessing import image


def train_model(blur):
    generated_train_data = ImageGenerator(blur)

    # Layers before the branches
    i1 = Input(shape=(20, 200, 1))
    conv_layer = Conv2D(filters=32, kernel_size=(4, 4), padding='same', kernel_initializer='normal')(i1)
    max_pool = MaxPool2D((2, 2))(conv_layer)

    conv_layer_2 = Conv2D(filters=64, kernel_size=(4, 4), padding='same', kernel_initializer='normal')(max_pool)
    max_pool_2 = MaxPool2D((2, 2))(conv_layer_2)

    conv_layer_3 = Conv2D(filters=128, kernel_size=(4, 4), padding='same', kernel_initializer='normal')(max_pool_2)
    max_pool_3 = MaxPool2D((2, 2))(conv_layer_3)



    flatten_layer = Flatten()(max_pool_3)
    dropout_3 = Dropout(0.3)(flatten_layer)
    dense_layer = Dense(10 * 26, activation=tf.nn.relu)(dropout_3)

    outputs=[]
    for i in range(10):
        # Layers in the branches
        b1 = Dense(len(string.ascii_uppercase), activation="softmax")(dense_layer)
        outputs.append(b1)

    string_model = Model(inputs=i1, outputs=outputs)

    string_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    string_model.fit_generator(generated_train_data, validation_steps=350, epochs=50, steps_per_epoch=500)

    return string_model

def get_precision(model, blur):
    c = 0
    for i in range(1000):
        img = ImageGenerator(blur).create_image()
        real_img=img[1]
        #print(real_img)
        #img = image.load_img(img_path)
        img = (1 - np.array(img[0]).reshape(20, 200, 1)) / 255.0
        img=np.expand_dims(img, axis=0)
        pred = model.predict(img)
        letter_dict = [i for i in string.ascii_uppercase]
        word = ""
        for letter in pred:
            word=word+letter_dict[letter.argmax()]
        #print(word)
        if real_img==word:
            c=c+1
    return str(c)

output = []
for i in range(1, 6):
    #model = train_model(i)
    #model.save("model_"+str(i))
    model = load_model('model_'+str(i))
    output.append("Blur factor: "+str(i)+"--->"+get_precision(model, i))

for i in output:
    print(i)
