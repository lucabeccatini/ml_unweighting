import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# # # # # logarithm of the weights in order to avoid numerical problems ???

# # # # # unwgting class ???

# # # # # implement for negative wgt


##################################################
# evaluate the weight of events
##################################################

# reading the momenta and weight of events
X_train, X_val, wgt_train, wgt_val = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
with open('/home/lb_linux/nn_unwgt/info_wgt_events.txt', 'r') as infof:
    data = np.empty(0)         # # # use np.empty(shape) to avoid vstack
    for line in infof.readlines():
        if (len(data) == 0):
            data = np.copy([float(i) for i in line.split()])
        else:
            data = np.vstack([data, [float(i) for i in line.split()] ])
    X_train, X_val = data[:-1000, :-1], data[-1000:, :-1]
    wgt_train, wgt_val = data[:-1000, -1], data[-1000:, -1]

# define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape = (6,)),
    tf.keras.layers.Dense(16, activation='relu'), 
    tf.keras.layers.Dense(1)
    ])

# loss function and compile the model 
model.compile(optimizer='adam', loss='mean_squared_error')

# training and test
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
history = model.fit(X_train, wgt_train, validation_data=(X_val, wgt_val), batch_size=1000, epochs=100, callbacks=[callback]) 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()



##################################################
# unweighting
##################################################

# definitions of maximum functions
def max_quantile_reduction(arr):
    arr = np.sort(arr)
    tot_sum = np.sum(arr)
    part_sum = 0 
    rem_cont = 0.001                                       # coefficient of the remaining contribution to the total sum
    for i in len(arr):
        part_sum += arr[i]
        if (part_sum > tot_sum*(1-rem_cont)):
            max = arr[i]
            break
    return max

# # # def max_median_reduction(arr):

my_max = max_quantile_reduction

np.random.seed(7)

# first unweighting
s1 = model.predict(X_val)
s1_max = my_max(s1)
rand1 = np.random.rand(len(s1))
s2 = []
for i in range(len(s1)):
    if (s1[i]/s1_max > rand1[i]):

