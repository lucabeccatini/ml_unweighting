import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import time 


##################################################
# settings
##################################################
seed_all = 2
norm = "cmass"                                   # s_pro, s_int or cmass
output = "lnw"                                   # wgt or lnw
lossfunc = "mse"                                 # mse or chi
maxfunc = "mqr"                                  # mqr or mmr
unwgt = "new"                                    # new or pap

eff1_st = 0.099                                  # standard effeciencies for the first unwgt
eff2_st = 0.997                                  # standard effeciencies for the second unwgt
E_cm_pro = 13000                                 # energy of cm of protons
t_ratio = 0.002       # t_surrogate / t_standard = [1/20, 1/50, 1/100, 1/500], t: time to compute one event

# efficiencies functions
def f_kish(z):                                   # kish factor
    if (len(z)==0):
        return 0
    res = np.sum(z)**2 / (len(z) * np.sum(z**2))
    return res

def efficiency(z_f, s_i, s_i_max):                 # efficiency of the unwgt 
    #eff = f_kish(z_f) * np.sum(s_i) / (len(s_i) * s_i_max)
    # eff given by the sum of the probability to keep each event, to avoid fluctuations
    eff = np.sum(np.minimum(np.abs(s_i), s_i_max)) / (len(s_i)*s_i_max)
    return eff 

def effective_gain(z_f, eff1, eff2):                           # effective gain factor
    # eff gain given by the ratio between the total time T of the standard method and the surrogate one to obtain the same results
    res = f_kish(z_f) / (t_ratio*(eff1_st*eff2_st)/(eff1*eff2) + (eff1_st*eff2_st/eff2))
    return res



##################################################
# prediction of weights
##################################################

# reading the momenta and weight of events
data = np.empty(0)         # # # use np.empty(shape) to avoid vstack
with open('/home/lb_linux/nn_unwgt/info_wgt_events_5iter.txt', 'r') as infof:
    # data in info: px_t, py_t, pz_t, E_t, pz_tbar, E_tbar, wgt 
    for line in infof.readlines():
        if (len(data) == 0):
            data = np.copy([float(i) for i in line.split()])
        else: 
            data = np.vstack([data, [float(i) for i in line.split()] ])


# input normalization
def energy_cm(X):
    res = np.sqrt((X[:, 3]+X[:, 5])**2 - (X[:, 2]+X[:, 4])**2)
    return res

def beta(X):                                     # beta of top-antitop in the lab frame
    res = np.abs(X[:, 2]+X[:, 4]) / (X[:, 3]+X[:, 5])
    return res

def rapidity(X):                                 # rapidity of top in the lab frame
    res = 0.5 * np.log((X[:, 3]+X[:, 2]) / (X[:, 3]-X[:, 2]))
    return res

#X_train, X_val = np.empty(0), np.empty(0)
if (norm=="s_pro"):
    X_data = data[:, :-1] / E_cm_pro 
    X_train, X_val = X_data[:-4000, :], X_data[-4000:, :]
if (norm=="s_int"):
    E_cm_int = energy_cm(data[:, :-1])
    X_data = data[:, :-1] / E_cm_int 
    X_train, X_val = X_data[:-4000, :], X_data[-4000:, :]
if (norm=="cmass"):
    E_cm_int = energy_cm(data[:, :-1])
    beta_int = beta(data[:, :-1])
    X_data = np.empty(shape=data[:, :-2].shape)
    X_data[:, 0] = E_cm_int / E_cm_pro 
    X_data[:, 1] = rapidity(data[:, :-1])
    X_data[:, 2] = data[:, 0] / E_cm_int
    X_data[:, 3] = data[:, 1] / E_cm_int 
    X_data[:, 4] = (-(beta_int/np.sqrt(1-beta_int**2))*data[:, 3] + (1/np.sqrt(1-beta_int**2))*data[:, 2]) / E_cm_int
    X_train, X_val = X_data[:-4000, :], X_data[-4000:, :]


# output inizialization
wgt_train, wgt_val = data[:-4000, -1], data[-4000:, -1]


# define the model
tf.random.set_seed(seed_all) 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape = (len(X_train[0]), )),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
    ])


# loss function and compile the model 
def chi_sqaure(w_true, w_pred):
    chisq = 0
    chisq = (w_pred - w_true)**2 / w_true
    return chisq

if (lossfunc=="mse"):
    loss = 'mean_squared_error'
if (lossfunc=="chi"):
    loss = chi_sqaure
model.compile(optimizer='adam', loss=loss)   # metrics=[f_eff]


# training and test
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
if (output=="wgt"):
    history = model.fit(X_train, wgt_train, validation_data=(X_val, wgt_val), batch_size=1000, epochs=100, callbacks=[callback])         # predict w
if (output=="lnw"):
    history = model.fit(X_train, np.abs(np.log(wgt_train)), validation_data=(X_val, np.abs(np.log(wgt_val))), batch_size=1000, epochs=100, callbacks=[callback])         # predict the abs(log(w))
plt.plot(history.history['loss'], label="Training")
plt.plot(history.history['val_loss'], label="Validation")
plt.title('model loss')
plt.yscale('log')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.savefig("/home/lb_linux/nn_unwgt/plot_16_16_{}_{}_{}_{}_seed{}_train.pdf".format(norm, lossfunc, output, unwgt, seed_all), format='pdf')



##################################################
# definitions of maxima functions
##################################################

def max_quantile_reduction(array_s, array_x=[]):
    # define a reduced maximum such that the overweights' remaining contribution to the total sum of weights is lower or equal to r*total sum
    part_sum = 0 
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
    arr_r = [0.1, 0.01, 0.001, -1, -9] 
    #arr_r = [0.7, 0.5, 0.3, 0.1] 

if (maxfunc=="mmr"):
    my_max = max_median_reduction
    arr_r = [100, 50, 10, 5, 0, -1, -9]

if (unwgt=="new"):
    arr_smax = np.empty(len(arr_r))
if (unwgt=="pap"):
    arr_wmax = np.empty(len(arr_r))
arr_eff1 = np.empty(len(arr_r))



##################################################
# preparation of plots
##################################################

fig_eff, axs_eff = plt.subplots(5, figsize=(10, 15))  
if(unwgt=="new"):
    axs_eff[0].set(xlabel="s_max", ylabel="eff_1")
if(unwgt=="pap"):
    axs_eff[0].set(xlabel="w_max", ylabel="eff_1")
axs_eff[1].set(xlabel="x_max", ylabel="eff_2")
#axs_eff[2].set(xlabel="s_max", ylabel="x_max")
plot_legend = """Layers: 6, 16, 16, 1 \nEpochs: 100 \nBatch size: 1000 \nEv_train: 12000 \nEv_val: 4000
Normalization: {} \nLoss: {} \nOutput: {} \nMax func: {} \nUnwgt: {} \nSeed tf and np: {}""".format(norm, lossfunc, output, maxfunc, unwgt, seed_all)
fig_eff.legend(title = plot_legend)

fig_ws, axs_ws = plt.subplots(3, figsize=(8, 9))
axs_ws[0].set(xlabel="w/s", ylabel="dN/d(w/s)")
axs_ws[1].set(xlabel="w/s", ylabel="dN/d(w/s)")
axs_eff[2].set_title(label="f_eff")
axs_ws[2].set(xlabel="w", ylabel="w/s")
fig_ws.legend(title = plot_legend)

fig_zk, axs_zk = plt.subplots(len(arr_r)*len(arr_r), figsize=(6, 36))
fig_zk.legend(title = plot_legend)


##################################################
# unweighting
##################################################

mtx_xmax, mtx_eff2, mtx_kish, mtx_feff = np.empty((len(arr_r), len(arr_r))), np.empty((len(arr_r), len(arr_r))), np.empty((len(arr_r), len(arr_r))), np.empty((len(arr_r), len(arr_r)))

s1 = model.predict(X_val)
s1 = s1.reshape(len(s1))
if (output=="lnw"):
    s1 = np.e**(-s1)                         # model predict -log(w)
    
for i_r1 in range(len(arr_r)):                   # loop to test different maxima conditions

    np.random.seed(seed_all)                     # each test has the same seed
    r = arr_r[i_r1]                              # parameter of the maxima function for the first unwgt
    
    # first unweighting
    rand1 = np.random.rand(len(s1))              # random numbers for the first unwgt
    w2 = np.empty(0)                             # real wgt evaluated after first unwgt
    s2 = np.empty(0)                             # predicted wgt kept by first unwgt
    z2 = np.empty(0)                             # predicted wgt after first unwgt
    x2 = np.empty(0)                             # ratio between real and predicted wgt of kept events

    if (unwgt=="new"):                           # new method for the unweighting
        s_max = my_max(s1)
        arr_smax[i_r1] = s_max
        for i in range(len(s1)):                 # first unweighting, based on the predicted wgt
            if (np.abs(s1[i])/s_max > rand1[i]):
                s2 = np.append(s2, s1[i])
                wgt_z = np.sign(s1[i])*np.maximum(1, np.abs(s1[i])/s_max)      # kept event's wgt after first unwgt
                z2 = np.append(z2, wgt_z) 
                w2 = np.append(w2, wgt_val[i]) 
                x2 = np.append(x2, wgt_val[i]/np.abs(s1[i]))
        arr_eff1[i_r1] = efficiency(z2, s1, s_max)
    if (unwgt=="pap"):                           # paper method for the unwgt
        w_max = my_max(wgt_val)                  # unwgt done respect w_max
        arr_wmax[i_r1] = w_max
        for i in range(len(s1)):                 # first unwgt, based on the predicted wgt
            if (np.abs(s1[i])/w_max > rand1[i]):
                s2 = np.append(s2, s1[i])
                wgt_z = np.sign(s1[i])*np.maximum(1, np.abs(s1[i])/w_max)
                z2 = np.append(z2, wgt_z)
                w2 = np.append(w2, wgt_val[i])    
                x2 = np.append(x2, wgt_val[i]/np.abs(s1[i]))
        arr_eff1[i_r1] = efficiency(z2, s1, w_max)

    for i_r2 in range(len(arr_r)):               # to test all combinations of s_max and x_max 
        # second unweighting
        rand2 = np.random.rand(len(x2))
        r = arr_r[i_r2]                          # parameter of the maxima function for the second unwgt

        s3 = np.empty(0)                         # predicted wgt kept by second unwgt
        z3 = np.empty(0)                         # predicted wgt after second unwgt
        x3 = np.empty(0)
        ztot = np.empty(0)
        z3_0ow = np.empty(0)                     # final events with no overwgt
        z3_1ow = np.empty(0)                     # final events with overwgt only in the first unwgt (reabsorbed)
        z3_2ow = np.empty(0)                     # final events with overwgt only in the second unwgt 
        z3_12ow = np.empty(0)                    # final events with overwgt in both unwgt
        if (unwgt=="new"):
            if (maxfunc=="mqr"):
                x_max = my_max(s2, x2*np.abs(z2))
            if (maxfunc=="mmr"):
                x_max = my_max(x2*np.abs(z2)) 
            for i in range(len(s2)):                 # second unweighting
                if ((np.abs(z2[i])*x2[i]/x_max) > rand2[i]):
                    s3 = np.append(s3, s2[i])
                    wgt_z = np.sign(z2[i])*np.maximum(1, np.abs(z2[i])*x2[i]/x_max)
                    z3 = np.append(z3, wgt_z)
                    x3 = np.append(x3, x2[i])
                    if (z2[i]==1 and wgt_z ==1):
                        z3_0ow = np.append(z3_0ow, wgt_z)
                    if (z2[i]>1 and wgt_z ==1):
                        z3_1ow = np.append(z3_1ow, wgt_z)
                    if (z2[i]==1 and wgt_z >1):
                        z3_2ow = np.append(z3_2ow, wgt_z)
                    if (z2[i]>1 and wgt_z >1):
                        z3_12ow = np.append(z3_12ow, wgt_z)

            mtx_eff2[i_r1, i_r2] = efficiency(z3, x2, x_max)
            mtx_feff[i_r1, i_r2] = effective_gain(z3, arr_eff1[i_r1], mtx_eff2[i_r1, i_r2])

            # for zk plot only 
            mtx_kish[i_r1, i_r2] = f_kish(z3)

            axs_zk[i_r1*len(arr_r)+i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                color=['blue', 'yellow', 'orange', 'red'], label="""r1: {} \nr2:{} \nf_kish: {:.5f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} 
                \nN_12ow: {}""".format(arr_r[i_r1], arr_r[i_r2], mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
            axs_zk[i_r1*len(arr_r)+i_r2].set_yscale('log')
            axs_zk[i_r1*len(arr_r)+i_r2].legend(loc='best')
            axs_zk[i_r1*len(arr_r)+i_r2].set(xlabel="z3", ylabel="dN/dz3")

        if (unwgt=="pap"):
            if (maxfunc=="mqr"):
                x_max = my_max(s2, x2)
            if (maxfunc=="mmr"):
                x_max = my_max(x2) 
            for i in range(len(s2)):                 # second unweighting
                if ((x2[i]/x_max) > rand2[i]):
                    s3 = np.append(s3, s2[i])
                    wgt_z = np.maximum(1, x2[i]/x_max)
                    z3 = np.append(z3, wgt_z)
                    ztot = np.append(ztot, z2[i]*wgt_z)
                    x3 = np.append(x3, x2[i])
            mtx_eff2[i_r1, i_r2] = efficiency(z3, x2, x_max) 
            mtx_feff[i_r1, i_r2] = effective_gain(ztot, arr_eff1[i_r1], mtx_eff2[i_r1, i_r2])
        mtx_xmax[i_r1, i_r2] = x_max
        #axs_eff[1].annotate(arr_r[i_r2], (x_max, mtx_eff2[i_r1, i_r2]))



##################################################
# unweighting of standard sample with fixed kish factor
##################################################
cmap = plt.get_cmap('plasma')
colors = cmap(np.linspace(0, 1, len(arr_r))) 

def effective_gain_st(eff1, eff2, eff_st):       # effective gain factor where the standard method computes all the matrix elements
    t_ratio = 0.002
    res = 1 / (t_ratio*eff_st/(eff1*eff2) + eff_st/eff2)
    return res

# unweighting of the standard sample with only the second unweighting to achieve the same kish factor of the surrogate method for a better comparison
mtx_eff_st, mtx_feff_st =  np.zeros((len(arr_r), len(arr_r))), np.zeros((len(arr_r), len(arr_r)))
s_st2 = s1                                       # no first unweighting
w_st2 = wgt_val 
x_st2 = w_st2 / np.abs(s_st2)
rand_st = np.random.rand(len(s_st2))
for i_r1 in range(len(arr_r)):
    for i_r2 in range(len(arr_r)):
        if (mtx_kish[i_r1, i_r2]<1):             # if kish_su<1, we require r2_st such that kish_st=kish_su
            kish_st = 0 
            r = (1 - mtx_kish[i_r1, i_r2]) * 2   # 
            for j in range(20): 
                x_max = my_max(s_st2, x_st2)
                s_st3 = np.empty(0)
                z_st3 = np.empty(0)
                x_st3 = np.empty(0)
                for i in range(len(s_st2)):                 # second unweighting
                    if ((x_st2[i]/x_max)>rand_st[i] and (x_st2[i]/x_max)<10):
                        s_st3 = np.append(s_st3, s_st2[i])
                        wgt_z = np.sign(s_st2[i])*np.maximum(1, x_st2[i]/x_max)
                        z_st3 = np.append(z_st3, wgt_z)
                        x_st3 = np.append(x_st3, x_st2[i])
                        kish_st = f_kish(z_st3)
                if (np.abs(mtx_kish[i_r1, i_r2]-kish_st)/mtx_kish[i_r1, i_r2] <= 0.01):
                    break
                else:                            # if kish_st!=kish_su we modify r and perform again the second unweighting
                    r += (kish_st - mtx_kish[i_r1, i_r2])
        else:                                    # if kish_su=1 we use the r2_st=r2_su
            r = arr_r[i_r2]
            x_max = my_max(s_st2, x_st2)
            s_st3 = np.empty(0)
            z_st3 = np.empty(0)
            x_st3 = np.empty(0)
            for i in range(len(s_st2)):                 # second unweighting
                if ((x_st2[i]/x_max) > rand_st[i]):
                    s_st3 = np.append(s_st3, s_st2[i])
                    wgt_z = np.sign(s_st2[i])*np.maximum(1, x_st2[i]/x_max)
                    z_st3 = np.append(z_st3, wgt_z)
                    x_st3 = np.append(x_st3, x_st2[i])
        mtx_eff_st[i_r1, i_r2] = efficiency(z_st3, x_st2, x_max)
        mtx_feff_st[i_r1, i_r2] = effective_gain_st(arr_eff1[i_r1], mtx_eff2[i_r1, i_r2], mtx_eff_st[i_r1, i_r2])
    axs_eff[3].plot(mtx_kish[i_r1], mtx_feff_st[i_r1], marker='.', color=colors[i_r1], label="s_max = {:.4f}".format(arr_smax[i_r1])) 
axs_eff[3].set(xlabel="Kish factor", ylabel="f_eff_st")
axs_eff[3].legend
axs_eff[3].legend(loc='best')


# plot of ovR (1/R) 
# x axis label with r1 and r2
x_ovR = ["{}\n{}".format(arr_r[i//len(arr_r)], arr_r[i%len(arr_r)]) for i in range(len(arr_r)*len(arr_r))]      

# y axis values with 1/R
y_ovR = [ 1 / ((arr_eff1[i//len(arr_r)] * mtx_eff2[i//len(arr_r), i%len(arr_r)] / mtx_eff_st[i//len(arr_r), i%len(arr_r)]) - arr_eff1[i//len(arr_r)]) for i in range(len(arr_r)*len(arr_r))] 

# bar color with the kish factor
kish_dif = np.max(mtx_kish) - np.min(mtx_kish) 
c_ovR = [cmap((mtx_kish[i//len(arr_r), i%len(arr_r)]-np.min(mtx_kish))/kish_dif) for i in range(len(arr_r)*len(arr_r))]
norm_ovR = mpl.colors.Normalize(vmin=np.min(mtx_kish), vmax=np.max(mtx_kish)) 
sm_ovR = plt.cm.ScalarMappable(cmap=cmap, norm=norm_ovR)
sm_ovR.set_array([])

axs_eff[4].bar(x_ovR, y_ovR, color=c_ovR)
x_ovr_min, x_ovr_max = axs_eff[4].get_xlim()
axs_eff[4].hlines(y=t_ratio, xmin=x_ovr_min, xmax=x_ovr_max, label="1/R", color='green')
plt.colorbar(sm_ovR, ax=axs_eff[4], ticks=np.linspace(np.min(mtx_kish), np.max(mtx_kish), 5))

axs_eff[4].set_xlabel("r1\nr2")
axs_eff[4].set_yscale('log')
axs_eff[4].set_ylabel("1/((eff_1*eff_2/eff_st)-eff_1)")
axs_eff[4].legend(loc='best')


"""
# st unweighting performed respect w_i/w_max instead of x_i/x_max
mtx_eff_st, mtx_feff_st =  np.zeros((len(arr_r), len(arr_r))), np.zeros((len(arr_r), len(arr_r)))
w_st2 = wgt_val                        # no first unweighting
rand_st = np.random.rand(len(w_st2))
for i_r1 in range(len(arr_r)):         # this unweighting is performed over w
    for i_r2 in range(len(arr_r)):
        if (mtx_kish[i_r1, i_r2]<1):             # if kish_su<1, we require r2_st such that kish_st=kish_su
            kish_st = 0 
            r = (1 - mtx_kish[i_r1, i_r2]) * 4
            for j in range(20):
                w_max = my_max(w_st2)
                w_st3 = np.empty(0)
                z_st3 = np.empty(0)
                for i in range(len(w_st2)):                 # standard unweighting
                    if ((w_st2[i]/w_max) > rand_st[i]):
                        w_st3 = np.append(w_st3, w_st2[i])
                        wgt_z = np.sign(w_st2[i])*np.maximum(1, w_st2[i]/w_max)
                        z_st3 = np.append(z_st3, wgt_z)
                kish_st = f_kish(z_st3)
                if (np.abs(mtx_kish[i_r1, i_r2]-kish_st)/mtx_kish[i_r1, i_r2] <= 0.01):
                    break
                else:
                    r += (kish_st - mtx_kish[i_r1, i_r2])*8
        else:                                    # if kish_su=1, we require same eff2 for st and su
            r = arr_r[i_r2]
            w_max = my_max(w_st2)
            w_st3 = np.empty(0)
            z_st3 = np.empty(0)
            for i in range(len(w_st2)):                 # second unweighting
                if ((w_st2[i]/w_max) > rand_st[i]):
                    w_st3 = np.append(w_st3, w_st2[i])
                    wgt_z = np.sign(w_st2[i])*np.maximum(1, w_st2[i]/w_max)
                    z_st3 = np.append(z_st3, wgt_z)
        mtx_eff_st[i_r1, i_r2] = efficiency(z_st3, w_st2, w_max)
"""


##################################################
# plot of results
##################################################

if (unwgt=="new"):
    axs_eff[0].plot(arr_smax, arr_eff1, marker='.')
if (unwgt=="pap"):
    axs_eff[0].plot(arr_wmax, arr_eff1, marker='.')
axs_eff[0].legend(loc='best')
axs_eff[1].set_xlim([0, 12])
for i_r1 in range(len(arr_r)):                   # to mark the values for max=real_max with larger points
    #axs_eff[0].annotate(arr_r[i_r1], xy=(arr_smax[i_r1], arr_eff1[i_r1]))
    if (arr_r[i_r1]==0):
        if (unwgt=="new"):
            axs_eff[0].scatter(arr_smax[i_r1], arr_eff1[i_r1], s= 50)
        if (unwgt=="pap"):
            axs_eff[0].scatter(arr_wmax[i_r1], arr_eff1[i_r1], s= 50)
    for i_r2 in range(len(arr_r)):
        if (arr_r[i_r1]==0):
            axs_eff[1].scatter(mtx_xmax[i_r1, i_r2], mtx_eff2[i_r1, i_r2], s=50)
        else:
            if (arr_r[i_r2]==0):
                axs_eff[1].scatter(mtx_xmax[i_r1, i_r2], mtx_eff2[i_r1, i_r2], s=50)
if (unwgt=="new"):
    for i in range(len(arr_r)):                      # plot the curve of the efficiencies in function of the x_max with fixed s_max
        axs_eff[1].plot(mtx_xmax[i, :], mtx_eff2[i], marker='.', color=colors[i], label="s_max = {:.4f}".format(arr_smax[i]))
if (unwgt=="pap"):
    for i in range(len(arr_r)):                      # plot the curve of the efficiencies in function of the x_max with fixed s_max
        axs_eff[1].plot(mtx_xmax[i, :], mtx_eff2[i], marker='.', color=colors[i], label="w_max = {:.4f}".format(arr_wmax[i]))
axs_eff[1].legend
axs_eff[1].legend(loc='best')

for i in range(len(arr_r)):                      # x, y labels of the f_eff colormap
    if (unwgt=="new"):
        axs_eff[2].text(0+i, -0.7 , "s_max: {:.4f} \nr: {}".format(arr_smax[i], arr_r[i]), fontsize = 6)
    if (unwgt=="pap"):
        axs_eff[2].text(0+i, -0.7 , "w_max: {:.4f} \nr: {}".format(arr_wmax[i], arr_r[i]), fontsize = 6)
    axs_eff[2].text(-1, 0+i , "x_max*: {:.3f} \nr: {}".format(mtx_xmax[i, i], arr_r[i]), fontsize = 6)
#axs_eff[2].axis('off')
axs_eff[2].set_yticklabels([])
axs_eff[2].set_xticklabels([])
h1 = axs_eff[2].pcolormesh(mtx_feff, cmap=cmap)
plt.colorbar(h1, ax=axs_eff[2])

x1 = np.abs(np.divide(wgt_val, s1))
lin_bins1 = np.linspace(min(x1), max(x1), 10**3) 
lin_bins2 = np.linspace(-0.5, 2.5, 50) 
axs_ws[0].hist(x1, bins=lin_bins1)
axs_ws[1].hist(x1, bins=lin_bins2)
wbins = np.linspace(min(wgt_val), max(wgt_val), 50)
xbins = np.linspace(0, 2, 20)
h2 = axs_ws[2].hist2d(wgt_val, x1, bins=[wbins, xbins])
plt.colorbar(h2[3], ax= axs_ws[2]) 

fig_eff.savefig("/home/lb_linux/nn_unwgt/plot_16_16_{}_{}_{}_{}_{}_seed{}_eff.pdf".format(norm, lossfunc, output, maxfunc, unwgt, seed_all), format='pdf')
fig_ws.savefig("/home/lb_linux/nn_unwgt/plot_16_16_{}_{}_{}_{}_{}_seed{}_ws.pdf".format(norm, lossfunc, output, maxfunc, unwgt, seed_all), format='pdf')
fig_zk.savefig("/home/lb_linux/nn_unwgt/plot_16_16_{}_{}_{}_{}_{}_seed{}_zk.pdf".format(norm, lossfunc, output, maxfunc, unwgt, seed_all), format='pdf')



print("\n--------------------------------------------------")
print(plot_legend)