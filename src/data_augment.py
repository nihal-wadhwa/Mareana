from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os

folder = '/Users/dhruv/Desktop/final_symbols/caution'
symbols = [os.path.join(folder,f) for f in os.listdir(folder)]
datagen = ImageDataGenerator(
                rotation_range=360,
                rescale=1./255,
                shear_range=0.15,
                zoom_range=0.05,
                fill_mode='nearest')

for symbol in symbols:
        img = cv2.imread(symbol)
        img = cv2.resize(img, (64, 64))   
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='../train/caution', save_prefix='caution', save_format='png'):
                i += 1
                if i > 7:
                        break  # otherwise the generator would loop indefinitely