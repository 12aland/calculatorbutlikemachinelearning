import tensorflow as tf
import numpy as np
import random
from tensorflow import keras
from tensorflow.python.keras.layers import Concatenate

xslist = []
yslist = []
#zslist = []
for x in np.arange(-2000.0,2001.0,1.0):
    p = random.random() * 10
    q = random.random() * 10
    xslist.append(p)
    xslist.append(q)
    yslist.append(p+q)
    #zslist.append(2*x+2)

#xslist = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
print(xslist)
print(yslist)
#print(zslist)
#yslist = [-4.0, -1.0, 0.0, 1.0, 4.0, 9.0, 16.0]
model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, input_shape=[2]),
  #tf.keras.layers.Dense(128),
  #tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
xs = np.array(xslist, dtype=float)
xs.shape = (int(len(xslist)/2),2)
#ys = np.array(yslist, dtype=float)
ys = np.array(yslist, dtype=float)
model.fit(xs, ys, epochs=2000)
r = np.array([[1.0],[2.0]], dtype=float)
r.shape = (1,2)
print(model.predict(r))
r = np.array([[-2.0],[3.0]], dtype=float)
r.shape = (1,2)
print(model.predict(r))
r = np.array([[34.0],[1.0]], dtype=float)
r.shape = (1,2)
print(model.predict(r))

r = np.array([[334.323],[23.56]], dtype=float)
r.shape = (1,2)
print(model.predict(r))