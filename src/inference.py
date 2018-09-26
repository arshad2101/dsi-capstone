from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
# dimensions of our images
img_width, img_height = 256, 256

# load the model we saved
model = load_model('../models/DR_Five_Classes_recall_0.6358.h5')
#model.compile(loss='binary_crossentropy',
 #             optimizer='rmsprop',
  #            metrics=['accuracy'])


image_path = '../sample/'
# predicting images]
for img_name in os.listdir(image_path):  # no need to convert to np.array yet...
    img = image.load_img(image_path+img_name, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x_new = np.expand_dims(x, axis=0)
    images = np.vstack([x_new])
    prob = model.predict_on_batch(images)
    classes = model.predict_classes(images)

    print('\n')

    if(classes[0] == 0):
        print("No DR")
        plt.imshow(img)
        plt.show()
        print(img_name)
    if(classes[0] == 1):
        print("Mild")
        plt.imshow(img)
        plt.show()
        print(img_name)
    if(classes[0] == 2):
        print("Moderate")
        plt.imshow(img)
        plt.show()
        print(img_name)
    if(classes[0] == 3):
        print("Severe")
        plt.imshow(img)
        plt.show()
        print(img_name)
    if(classes[0] == 4):
        print("Proliferative DR")
        plt.imshow(img)
        plt.show()
        print(img_name)


    print("No DR=",prob[0][0]," ","Mild=",prob[0][1],"Moderate=",prob[0][2],"Severe=",prob[0][3],"Proliferative DR=",prob[0][4])
