from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
import skimage


folder = 'symbols/sterile.png/'
symbols = [os.path.join(folder,f) for f in os.listdir(folder)]
datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rescale=1./255,
                shear_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                fill_mode='nearest')

for symbol in symbols[:5]:
        img = cv2.imread(symbol)
        #img = load_img('symbols/caution_3.png/25_59_9_caution_3.png')
        img = cv2.resize(img, (48, 48))   
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='train/sterile', save_prefix='sterile', save_format='png'):
                i += 1
                if i > 2:
                        break  # otherwise the generator would loop indefinitely