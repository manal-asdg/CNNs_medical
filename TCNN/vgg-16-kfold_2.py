from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, BatchNormalization, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import kflod
import numpy as np
from keras.preprocessing.image import array_to_img
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)


def randomColor(image):  # NOT USE FOR NOW
    image = array_to_img(image)
    random_factor = np.random.randint(0, 31) / 10.
    # color_image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(-20, 21) / 10.
    brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
    random_factor = np.random.randint(0, 31) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)


cvscores = []
for i in range(2, 3):  # 5 folds
    train_imgs, train_labels, test_imgs, test_labels = kflod.load_data(i)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=20,
        height_shift_range=20,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest'

    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255

    )

    train_generator = train_datagen.flow(
        train_imgs, train_labels,
        batch_size=32, shuffle=True)

    validation_generator = test_datagen.flow(
        test_imgs, test_labels,
        batch_size=32)


    def add_new_last_layer(base_model):
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)  # new FC layer
        # x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(4096, activation='relu')(x)
        # x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        predictions = Dense(2, activation='softmax')(x)  # new softmax layer

        model = Model(inputs=base_model.input, outputs=predictions)
        return model


    def setup_to_transfer_learn(model, base_model):
        """Freeze all layers and compile the model"""
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))  # download the model from the internet
    model = add_new_last_layer(base_model)  # add new layer to the model
    setup_to_transfer_learn(model, base_model)

    history = model.fit_generator(
        train_generator,
        epochs=20,
        validation_data=validation_generator)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False,
                              embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    '''
    unfreeze last block
    '''
    for layer in model.layers[:15]:
        layer.trainable = False
    for layer in model.layers[15:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    history2 = model.fit_generator(
        train_generator,
        epochs=100,
        validation_data=validation_generator, callbacks=[tensorboard])
    scores = model.evaluate(test_imgs / 255, test_labels, verbose=0)
    cvscores.append(scores[1] * 100)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    plt.plot(history2.history['acc'])
    plt.plot(history2.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
for i in cvscores:
    print(i)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }

pre = model.predict(test_imgs / 255)[:, 1]
true = test_labels[:, 1]
with open("vgg16_auc.txt", 'w') as f:
    for (y, x) in zip(pre, true):
        f.write(str(x) + ' ' + str(y) + '\n')

fpr, tpr, threshold = roc_curve(true, pre)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='CNN-V curve (area = %0.5f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity', fontsize=18)
plt.ylabel('Sensitivity ', fontsize=18)
plt.legend(loc="lower right", prop=font1)
plt.show()
