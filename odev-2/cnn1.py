from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
from keras import layers
from keras import ops

model = keras.Sequential()
model.add(keras.Input(shape=(100, 100, 1)))  # 100x100x1 RGB görüntü
model.add(layers.Conv2D(10, 2, strides=1,use_bias=False, activation="relu"))
model.add(layers.Conv2D(2, 2, strides=2,use_bias=False,activation="relu"))
model.add(layers.MaxPooling2D(3))


model.summary()



# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(3))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.Conv2D(32, 3, activation="relu"))
# model.add(layers.MaxPooling2D(2))
#
# # And now?
# model.summary()
#
# # Now that we have 4x4 feature maps, time to apply global max pooling.
# model.add(layers.GlobalMaxPooling2D())
#
# # Finally, we add a classification layer.
# model.add(layers.Dense(10))