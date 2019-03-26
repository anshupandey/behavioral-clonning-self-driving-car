
# Loading the packages
import pandas
import cv2
import numpy
import matplotlib.pyplot as plt

# loading the csv file using pandas.
df = pandas.read_csv(r"data/driving_log.csv",header=None)
# checking the first five rows
df.head()

#checking the shape of data
df.shape

# creating list for storing the images and the respective measurements
images = []
measurements = []
# iterating through the dataframe to read every image one by one
for i in range(df.shape[0]):
    #extracting the steering angle value from 3rd column
    steering_center = df[3][i]
    # creating a correction factor for left and right images
    correction = 0.3 # this parameter is tuned 5 times to get this value
    # using the correction factor getting the measurement for left and right images
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    # image path for center image
    center_path = df[0][i]
    #loading image using opencv
    center_img = cv2.imread(center_path)
    # converting the image from BGR to the RGB format
    center_img = cv2.cvtColor(center_img,cv2.COLOR_BGR2RGB)
    # image path for the left image
    left_path = df[1][i]
    # loading the image using opencv
    left_img = cv2.imread(left_path)
    # converting the image from BGR to RGB using opencv
    left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB)
    # image path for the right image
    right_path = df[2][i]
    #loading the image
    right_img = cv2.imread(right_path)
    # converting into the RGB format from the BGR format
    right_img = cv2.cvtColor(right_img,cv2.COLOR_BGR2RGB)
    # appending the images    
    images.extend((center_img,left_img,right_img))
    # appending the respective measurements
    measurements.extend((steering_center,steering_left,steering_right))

#checking the total number of images
len(images)

# image augementation using flipping the images and reversing the sign of the respective measurement
aug_images,aug_measurements = [], []
for i in range(len(images)):
    aug_images.append(images[i])
    aug_measurements.append(measurements[i])
    aug_images.append(cv2.flip(images[i],1))
    aug_measurements.append(measurements[i]*-1.0)

# converting the images and measurements to numpy array format
xtrain = numpy.array(aug_images)
ytrain = numpy.array(aug_measurements)

print(xtrain.shape)
print(ytrain.shape)

from keras import models,layers

# creating the mdoel
model = models.Sequential()
# adding the lambda function layer to scale the image pixel values
model.add(layers.Lambda(lambda x:x/255.0 - 0.5,input_shape=(160,320,3)))
#adding the layer to crop the images by 70 from top and 20 from bottom
model.add(layers.Cropping2D(cropping=((70,20), (0,0))))
# adding the 5 convolutional layers 
model.add(layers.Conv2D(filters = 20, kernel_size=(5,5),activation='relu',strides=(2, 2)))
model.add(layers.Conv2D(filters = 40, kernel_size=(5,5),activation='relu',strides=(2, 2)))
model.add(layers.Conv2D(filters = 50, kernel_size=(5,5),activation='relu',strides=(2, 2)))
model.add(layers.Conv2D(filters = 60, kernel_size=(3,3),activation='relu'))
model.add(layers.Conv2D(filters = 60, kernel_size=(3,3),activation='relu'))
model.add(layers.Flatten())
# adding the dense layers
model.add(layers.Dense(100,activation='relu'))
# drop out optimization
model.add(layers.Dropout(0.5))
# addint the dense layers
model.add(layers.Dense(50,activation='relu'))
model.add(layers.Dense(10,activation='relu'))
# the output layer
model.add(layers.Dense(1))

# compile the model, optimizer used - RMSProp, loss = MSE
model.compile(optimizer='RMSProp',loss='mse')
# train the model with 10 epochs and 20% validation split
history_object = model.fit(xtrain,ytrain,
                           validation_split=0.2,
                           shuffle=True,
                           epochs=10,
                           batch_size=128,
                           verbose=True)

# save the trained model
model.save('model.h5')

# analysing the loss function for train and validation data
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
