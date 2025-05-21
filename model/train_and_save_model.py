import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths
train_dir = r'C:\Users\Thanusree\Downloads\brain_tumor\Training'
val_dir = r'C:\Users\Thanusree\Downloads\brain_tumor\Testing'

# ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(299, 299), class_mode='categorical', batch_size=32)
val_data = val_gen.flow_from_directory(val_dir, target_size=(299, 299), class_mode='categorical', batch_size=32)

# Model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=5)


# Save model
model.save('Tumor_Xception_model.h5')
