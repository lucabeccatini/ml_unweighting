import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# # # # # logarithm of the weights in order to avoid numerical problems ???

# # # # # unwgting class ???

# # # # # better to use pandas to record all the info ???


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
tf.random.set_seed(1) 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape = (6,)),
    tf.keras.layers.Dense(16, activation='relu'), 
    tf.keras.layers.Dense(1)
    ])

# loss function and compile the model 
model.compile(optimizer='adam', loss='mean_squared_error')

# training and test
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
history = model.fit(X_train, wgt_train, validation_data=(X_val, wgt_val), batch_size=1000, epochs=1, callbacks=[callback]) 
"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
"""



##################################################
# unweighting
##################################################

# definitions of maximum functions
def max_quantile_reduction(array_s, array_x=[]):
    part_sum = 0 
    r = 0.001                             # coefficient of the remaining contribution to the total sum
    
    if (len(array_x) == 0):                      # max for s_i
        arr_s = np.sort(array_s)
        tot_sum = np.sum(arr_s)
        for i in range(len(arr_s)):
            part_sum += arr_s[i]
            if (part_sum >= tot_sum*(1-r)):
                max_s = arr_s[i]
                break
        return max_s
    
    else:                                          # max for x_i
        arr_s = [s for _,s in sorted(zip(array_x, array_s))]         # sort s_i respect to x_i for the second unwgt
        arr_x = np.sort(array_x)
        tot_sum = np.sum(arr_s*arr_x)
        for i in range(len(arr_s)):
            part_sum += arr_s[i]*arr_x[i]
            if (part_sum >= tot_sum*(1-r)):
                max_x = arr_x[i]
                break
        return max_x

# # # def max_median_reduction(arr):

my_max = max_quantile_reduction

np.random.seed(1)

# first unweighting
s1 = model.predict(X_val)
s_max = my_max(s1)
rand1 = np.random.rand(len(s1))            
w2_true = np.empty(0)                            # real wgt evaluated after first unwgt
s2 = np.empty(0)                                 # predicted wgt kept by first unwgt
for i in range(len(s1)):                         # first unweighting, based on the predicted wgt
    if (s1[i]/s_max > rand1[i]):
        s2 = np.append(s2, s1[i])
        w2_true = np.append(w2_true, wgt_val[i])
        # w2 = np.append(w2, max(1, s1[i]/s_max))
x2 = np.divide(w2_true, s2)

# second unweighting, based on the ratio x
x_max = my_max(s2, x2)
rand2 = np.random.rand(len(x2)) 
s3 = np.empty(0)                                 # wgt of kept event after second unweighting
x3 = np.empty(0)
for i in range(len(s2)):                         # second unweighting, based on the ratio x
    if ((x2[i]*max(1, s2[i]/s_max)/x_max) > rand2[i]):
        s3 = np.append(s3, s2[i])
        x3 = np.append(x3, x2[i])
        # w3 = np.append(w3, max(1, x[i]*max(1, s2[i]/s_max)/x_max)) 

# efficiencies of the unweightings
def efficiency_1(s1, s2, s_max):
    eff = np.sum(np.maximum(1, s2/s_max))**2 * np.sum(s1) / (len(s2) * np.sum(np.maximum(1, s2/s_max)**2) * len(s1) * s_max)
    return eff 

def efficiency_2(x2, x3, x_max, s3, s_max):
    eff = np.sum(np.maximum(1, x3*np.maximum(1, s3/s_max)/x_max))**2 * np.sum(x2) / (len(x3) * np.sum(np.maximum(1, x3*np.maximum(1, s3/s_max)/x_max)**2) * len(x2) * x_max)
    return eff

eff1 = efficiency_1(s1, s2, s_max)
eff2 = efficiency_2(x2, x3, x_max, s3, s_max)
print(eff1)
print(eff2)
