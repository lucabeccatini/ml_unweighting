import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


##################################################
# settings
##################################################
seed_all = 2
norm = "s_prot"                                  # s_prot or s_gluon
maxfunc = "mqr"                                  # mqr or mmr
output = "w"                                     # w or logw



##################################################
# evaluate the weight of events
##################################################

# reading the momenta and weight of events
X_train, X_val, wgt_train, wgt_val = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
with open('/home/lb_linux/nn_unwgt/info_wgt_events_5iter.txt', 'r') as infof:
    # data in info: px_t, py_t, pz_t, E_t, pz_tbar, E_tbar, wgt 
    data = np.empty(0)         # # # use np.empty(shape) to avoid vstack
    for line in infof.readlines():
        if (len(data) == 0):
            data = np.copy([float(i) for i in line.split()])
        else: 
            data = np.vstack([data, [float(i) for i in line.split()] ])
    X_train, X_val = data[:-4000, :-1], data[-4000:, :-1]
    wgt_train, wgt_val = data[:-4000, -1], data[-4000:, -1]

def energy_cm(X):
    res = np.sqrt((X[3]+X[5])**2 - X[0]**2 - X[1]**2 - (X[2]+X[4])**2)
    return res

# input normalization
if (norm=="s_prot"):
    norm_const = 13000                           # normalization costant equal to s of proton
    X_train, X_val = X_train/norm_const, X_val/norm_const
if (norm=="s_gluon"):
    for i in range(len(X_train)):                # normalize each moment using s^
        s_train = energy_cm(X_train[i])
        X_train[i] /= s_train
        if (i < len(X_val)):
            s_val = energy_cm(X_val[i])
            X_val[i] /= s_val

# define the model
tf.random.set_seed(seed_all) 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape = (6,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

# loss function and compile the model 
model.compile(optimizer='adam', loss='mean_squared_error')

# training and test
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
if (output=="w"):
    history = model.fit(X_train, wgt_train, validation_data=(X_val, wgt_val), batch_size=1000, epochs=100, callbacks=[callback])         # predict w
if (output=="logw"):
    history = model.fit(X_train, np.abs(np.log(wgt_train)), validation_data=(X_val, np.abs(np.log(wgt_val))), batch_size=1000, epochs=100, callbacks=[callback])         # predict the abs(log(w))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.savefig("/home/lb_linux/nn_unwgt/plot_train_16_16_{}_{}_seed{}.pdf".format(norm, output, seed_all), format='pdf')



##################################################
# preparation of plots
##################################################

fig_eff, axs_eff = plt.subplots(2, figsize=(8, 6))  
axs_eff[0].set(xlabel="s_max", ylabel="eff_1")
axs_eff[1].set(xlabel="x_max", ylabel="eff_2")
fig_eff.legend(title = """Layers: 6, 16, 16, 1 \nEpochs: 100 \nBatch size: 1000 \nEv_train: 12000 \nEv_val: 4000 
                            \nMax func: {} \nNormalization: {} \nSeed tf and np: {} \n Output: {}""".format(maxfunc, norm, seed_all, output))

fig_ws, axs_ws = plt.subplots(3, figsize=(12, 6))
axs_ws[0].set(xlabel="w/s", ylabel="dN/d(w/s)")
axs_ws[1].set(xlabel="w/s", ylabel="dN/d(w/s)")
axs_ws[2].set(xlabel="w", ylabel="w/s")
fig_ws.legend(title = """Layers: 6, 16, 16, 1 \nEpochs: 100 \nBatch size: 1000 \nEv_train: 12000 \nEv_val: 4000 
                            \nMax func: {} \nNormalization: {} \nSeed tf and np: {} \n Output: {}""".format(maxfunc, norm, seed_all, output))

arr_eff1, arr_eff2, arr_smax, arr_xmax = [], [], [], []



##################################################
# definitions of maximum functions
##################################################

def max_quantile_reduction(array_s, array_x=[]):
    # define a reduced maximum such that the overweights' remaining contribution to the total sum of weights is lower or equal to r*total sum
    part_sum = 0 
    r = arr_r[i_r]                               # fraction of the remaining contribution of larger wgts to the total sum
    if (len(array_x) == 0):                      # max for s_i
        max_s = 0
        if (r <= 0):                             # to test overwgt maxima
            max_s = max(array_s) * (1-r)
            return max_s
        else:
            arr_s = np.sort(array_s)             # sorted s_i to determine s_max
            tot_sum = np.sum(arr_s)
            for i in range(len(arr_s)):
                part_sum += arr_s[i]
                if (part_sum >= tot_sum*(1-r)):
                    max_s = np.abs(arr_s[i])
                    break
            return max_s        
    else:                                        # max for x_i
        max_x = 0                                # for x_i the total sum is given by the sum over x_i*s_i
        if (r <= 0):
            max_x = max(array_x) * (1-r)
            return max_x
        else:
            arr_s = [s for _,s in sorted(zip(array_x, array_s))]         # sort s_i respect to x_i for the second unwgt
            arr_x = np.sort(array_x)
            tot_sum = np.sum(arr_s*arr_x) 
            for i in range(len(arr_s)):
                part_sum += arr_s[i]*arr_x[i]
                if (part_sum >= tot_sum*(1-r)):
                    max_x = np.abs(arr_x[i])
                    break
            return max_x

def max_median_reduction(arr):
    # define a reduced maximum such that it is the median over the maxima of different unweighted samples
    if (r <= 0):                                 # to test overwgt maxima
        max_s = max(arr) * (1-r)
        return max_s
    else:
        n_max = 50                               # number of maxima used for the median
        arr = np.flip(np.sort(arr))              # reversed sorted input array
        arr_max = np.zeros(n_max)
        r = arr_r[i_r]
        max_r = arr[0] * r 
        for j1 in range(n_max):
            rand_max = np.random.rand(len(arr))
            for j2 in range(len(arr)):
                if (arr[j2] > (max_r*rand_max[j2])):
                    arr_max[j1] = arr[j2]
                    break
            if (arr_max[j1] == 0):
                arr_max[j1] = arr[0]
        res = np.median(arr_max)
        return res

if (maxfunc=="mqr"): 
    my_max = max_quantile_reduction
    arr_r = [0.1, 0.01, 0.001, 0.0001, 0, -0.5, -1] 

if (maxfunc=="mmr"):
    my_max = max_median_reduction
    arr_r = [100, 50, 10, 5, 2, 0, -0.5, -1]



##################################################
# unweighting
##################################################

# efficiencies functions
def f_kish(w):                         # kish factor
    if (len(w)==0):
        return 0
    res = np.sum(w)**2 / (len(w) * np.sum(w**2))
    return res

def efficiency_1(s1, s2, s_max): 
    eff = f_kish(np.sign(s2)*np.maximum(1, np.abs(s2)/s_max)) * np.sum(s1) / (len(s1) * s_max)
    return eff 

def efficiency_2(x2, x3, x_max, s3, s_max):
    eff = f_kish(np.sign(s3)*np.maximum(1, x3*np.maximum(1, np.abs(s3)/s_max)/x_max)) * np.sum(x2) / (len(x2) * x_max)
    return eff

"""
def f_eff(w2, w3):                     # effective gain factor
    t_ratio = [20, 50, 100, 500]       # t_standard / t_surrogate
    eff_st = 0.1                       # standard efficiency
""" 


for i_r in range(len(arr_r)):                    # loop to test different maxima conditions

    np.random.seed(seed_all)                     # each test has the same seed

    # first unweighting
    s1 = model.predict(X_val)
    s1 = s1.reshape(len(s1))
    if (output=="logw"):
        s1 = np.e**(-s1)                         # model predict -log(w)
    s_max = my_max(s1)
    rand1 = np.random.rand(len(s1))             
    w2 = np.empty(0)                             # real wgt evaluated after first unwgt
    s2 = np.empty(0)                             # predicted wgt kept by first unwgt
    for i in range(len(s1)):                     # first unweighting, based on the predicted wgt
        if (np.abs(s1[i])/s_max > rand1[i]):
            s2 = np.append(s2, s1[i])
            w2 = np.append(w2, wgt_val[i])
    x2 = np.divide(w2, np.abs(s2))
    
    # second unweighting, based on the ratio x
    rand2 = np.random.rand(len(x2))
    if (maxfunc=="mqr"):
        x_max = my_max(s2, x2)
    if (maxfunc=="mmr"):
        x_max = my_max(x2) 
    s3 = np.empty(0)                             # wgt of kept event after second unweighting
    x3 = np.empty(0)
    for i in range(len(s2)):                     # second unweighting, based on the ratio x
        if ((x2[i]*max(1, np.abs(s2[i])/s_max)/x_max) > rand2[i]):
            s3 = np.append(s3, s2[i])
            x3 = np.append(x3, x2[i])
    
    # data for plots
    arr_eff1.append(efficiency_1(s1, s2, s_max))
    arr_eff2.append(efficiency_2(x2, x3, x_max, s3, s_max))
    arr_smax.append(s_max)
    arr_xmax.append(x_max)
    axs_eff[0].annotate(arr_r[i_r], xy=(arr_smax[i_r], arr_eff1[i_r]))
    axs_eff[1].annotate(arr_r[i_r], (arr_xmax[i_r], arr_eff2[i_r]))    
    if (arr_r[i_r]==0):
        axs_eff[0].axvline(x=s_max, color='r')
        axs_eff[1].axvline(x=x_max, color='r')



##################################################
# plot of results
##################################################
axs_eff[0].plot(arr_smax, arr_eff1, marker='.')
axs_eff[1].plot(arr_xmax, arr_eff2, marker='.')
x1 = np.divide(wgt_val, s1)
lin_bins1 = np.linspace(min(x1), max(x1), 10**3) 
lin_bins2 = np.linspace(-0.5, 2.5, 50) 
axs_ws[0].hist(x1, bins=lin_bins1)
axs_ws[1].hist(x1, bins=lin_bins2)
wbins = np.linspace(min(wgt_val), max(wgt_val), 50)
xbins = np.linspace(0, 2, 20)
h = axs_ws[2].hist2d(wgt_val, x1, bins=[wbins, xbins])
plt.colorbar(h[3], ax= axs_ws[2]) 
fig_eff.savefig("/home/lb_linux/nn_unwgt/plot_eff_{}_16_16_{}_{}_seed{}.pdf".format(maxfunc, norm, output, seed_all), format='pdf')
fig_ws.savefig("/home/lb_linux/nn_unwgt/plot_ws_{}_16_16_{}_{}_seed{}.pdf".format(maxfunc, norm, output, seed_all), format='pdf')


