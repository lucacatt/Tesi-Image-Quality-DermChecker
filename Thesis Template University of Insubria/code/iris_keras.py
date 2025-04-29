# Iris Dataset
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Split Dataset
from sklearn.model_selection 
    import train_test_split
x_train, x_test, y_train, y_test = 
    train_test_split(x, y, test_size=0.33)

# Sequential Model
import tensorflow as tf
tf_layers = [
    tf.keras.layers.Flatten(input_shape=(4,)),
    tf.keras.layers.Dense(units=512,activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
]
model = tf.keras.models.Sequential(tf_layers)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

# Print Model
for l in model.layers:
		text += str(l.get_config()['name']) + str(l.get_config()) + '\n'
print(text)
out = str(text) 

# Train Tensorboard
from time import time
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))
model.fit(
    x_train, 
    y_train, 
    epochs=50, 
    verbose=0, 
    callbacks=[tensorboard], 
    validation_data=(x_test, y_test)
)

# Print
print('ENDED')
