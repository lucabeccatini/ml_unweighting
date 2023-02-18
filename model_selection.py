import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

norm_sc = MinMaxScaler()
stan_sc = StandardScaler()


dataset = "8t10to6"            # 7iter, 10to6, 8t10to6
channel = "G128"               # G128 , G304 
load_module = False
cluster = True

if (cluster==True):
    path_scratch = "/globalscratch/ucl/cp3/lbeccati"
if (cluster==False):
    path_scratch = "/home/lb_linux/nn_unwgt/{}".format(channel)



##################################################
# reading
##################################################

time_0 = time.time()

# reading the momenta and weight of events
if (dataset=="8t10to6"):
    n_data = 8*10**6
if (dataset=="10to6"):
    n_data = 10**6
if (dataset=="7iter"):
    n_data = 64000
data = np.empty((n_data, 33)) 
with open("{}/info_wgt_events_{}_{}.txt".format(path_scratch, channel, dataset), 'r') as infof:
    print("Start readind events")
    event = 0
    # in each line of info: (px, py, pz, E) of ga, gb; e-, e+, g1, g2, d, dbar and wgt 
    for line in infof.readlines():
        data[event, :] = [float(i) for i in line.split()]
        if (event%(n_data//100)==0):
            print("Events read: {}% in {:.2f}s".format((event/(n_data//100)), time.time()-time_0))
        event +=1
val_len = len(data)//4
train_len = (len(data)//(4)) * 3 + len(data)%(4)

time_read = time.time() - time_0 


##################################################
# settings
##################################################

lossfunc = "mse" 
#lossfunc = "hub" 
#lossfunc = "lch" 
#lossfunc = "chi" 

input = "ptitf" 

#output = "wno" 
output = "wst" 
#output = "lno" 
#output = "lst" 

if (output=="wno" or output=="lno"):
    delta_hub = 0.2
if (output=="wst" or output=="lst"):
    delta_hub = 3

seed_all = 2 

layers_set = ["4x64", "4x128", "8x64", "128to16"]
l_r_set = [0.01, 0.001, 0.0001]
batch_size_set = [32, 128, 1000]
output_activation_set = ["linear", "leaky_relu"]


with PdfPages("{}/model_selection_{}_{}_{}_{}_seed{}_{}.pdf".format(path_scratch, channel, input, output, lossfunc, seed_all, dataset)) as pdf: 

    col_lab = ["model", "val loss", "epochs", "eff_1 \neff_2", "alpha kish", "f_eff", "time_pred \ntime_unwgt1 \ntime_unwgt2"]
    row_max = 19
    count_table = 0

    for lays in layers_set:
        for l_r in l_r_set:
            for b_s in batch_size_set:
                for o_a in output_activation_set:
                    n_epochs = 1000 
                    learn_rate = l_r
                    batch_size = b_s
                    ratio_train_val = 3                              # number of training events over number of validation events

                    if (channel=="G128"):
                        eff1_st = 0.6832                                  # standard effeciencies for the first unwgt
                        eff2_st = 0.0205                                  # standard effeciencies for the second unwgt
                    if (channel=="G304"):
                        eff1_st = 0.6537                                  # standard effeciencies for the first unwgt
                        eff2_st = 0.0455                                  # standard effeciencies for the second unwgt
                    E_cm_pro = 13000                                 # energy of cm of protons
                    t_ratio = 0.002       # t_surrogate / t_standard = [1/20, 1/50, 1/100, 1/500], t: time to compute one event

                    part_name = ["e-", "e+", "g1", "g2", "d", "d~"]


                    # efficiencies functions
                    def f_kish(z):                                   # kish factor
                        if (len(z)==0):
                            return 0
                        res = np.sum(z)**2 / (len(z) * np.sum(z**2))
                        return res

                    def efficiency(s_i, s_i_max):                 # efficiency of the unwgt 
                        #eff = f_kish(z_f) * np.sum(s_i) / (len(s_i) * s_i_max)
                        # eff given by the sum of the probability to keep each event (to avoid fluctuations), s_i: initial weight
                        eff = np.sum(np.minimum(np.abs(s_i), s_i_max)) / (len(s_i)*s_i_max)
                        return eff 

                    def effective_gain(z_f, eff1, eff2):                           # effective gain factor
                        # eff gain given by the ratio between the total time T of the standard method and the surrogate one to obtain the same results
                        res = f_kish(z_f) / (t_ratio*(eff1_st*eff2_st)/(eff1*eff2) + (eff1_st*eff2_st/eff2))
                        return res


                    time_1 = time.time()

                    # input normalization
                    def energy_cm(X):
                        res = np.sqrt((X[:, 3]+X[:, 7])**2 - (X[:, 2]+X[:, 6])**2)       # E_cm = sqrt((E_g1+E_g2)**2-(pz_g1+pz_g2)**2)
                        return res

                    def beta(X):                                     # beta in the lab frame
                        res = (X[:, 2]+X[:, 6]) / (X[:, 3]+X[:, 7])      # beta = (pz_g1+pz_g2)/(E_g1+E_g2))
                        return res

                    def rapidity(X):                                 # rapidity in the lab frame
                        res = 0.5 * np.log((X[:, 3]+X[:, 2]) / (X[:, 3]-X[:, 2]))        # y = 1/2 * ln((E+pz)/(E-pz))
                        return res


                    if (input=="ptitf"):         # momentum transverse over E_int, phi, theta
                        input_name = ["E_int/E_pro", "beta", "p_T/E_int", "theta", "phi"]
                        E_cm_int = energy_cm(data[:, 0:8])
                        beta_int = beta(data[:, 0:8])
                        X_data = np.empty(shape=(len(data), 16))
                        X_data[:, 0] = E_cm_int / E_cm_pro 
                        X_data[:, 1] = beta_int
                        phi_d = np.arccos(data[:, 24] / np.sqrt(data[:, 24]**2+data[:, 25]**2)) * np.sign(data[:, 25])           # phi angle of d 
                        for i in range(5):       # for e-, e+, g3, g4, d
                            X_data[:, i*3+2] = np.sqrt(data[:, i*4+8]**2 + data[:, i*4+9]**2) / E_cm_int          # transverse momentum  in the cm over E_int
                            X_data[:, i*3+3] = np.arccos((-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11] + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10])
                            / np.sqrt(data[:, i*4+9]**2 + data[:, i*4+8]**2 + (-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11]
                            + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10])**2))                        # theta angle in parallel plane  (0, pi)
                            if (i<4):
                                X_data[:, i*3+4] = np.arccos(data[:, i*4+8] / np.sqrt(data[:, i*4+8]**2+data[:, i*4+9]**2)) * np.sign(data[:, i*4+9]) - phi_d            # phi angle in transverse plane  (-pi, pi)
                                X_data[:, i*3+4] -= (np.abs(X_data[:, i*3+4])//np.pi)*2*np.pi*np.sign(X_data[:, i*3+4])          # to rinormalize phi between (-pi, pi)
                        X_train, X_val = X_data[:train_len, :], X_data[-val_len:, :]

                    if (input=="p3pap"):
                        input_name = ["p_z(g_a)", "p_z(g_b)", "p_x", "p_y", "p_z"]
                        X_data = np.empty(shape=(len(data), 20))
                        X_data[:, 0] = data[:, 2]
                        X_data[:, 1] = data[:, 6]
                        for i in range(6):       # for e-, e+, g3, g4, d, d~
                            X_data[:, i*3+2] = data[:, i*4+8]                      # px
                            X_data[:, i*3+3] = data[:, i*4+9]                      # py 
                            X_data[:, i*3+4] = data[:, i*4+10]
                        X_data = X_data/(E_cm_pro/2)
                        X_train, X_val = X_data[:train_len, :], X_data[-val_len:, :]


                    # output inizialization

                    wgt_train, wgt_val = np.abs(data[:train_len, -1]), np.abs(data[-val_len:, -1])      # take the abs of wgt to avoid problems
                    wgt_train_pred, wgt_val_pred = np.empty(len(wgt_train)), np.empty(len(wgt_val))     # value to predict

                    if (output=="wno"):             # predict the wgt (or the wgt with a lower cut) normalized between 0 and 1
                        wgt_train_pred = wgt_train
                        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
                        norm_sc.fit(wgt_train_pred)
                        wgt_train_pred = norm_sc.transform(wgt_train_pred)
                        wgt_val_pred = wgt_val
                        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
                        wgt_val_pred = norm_sc.transform(wgt_val_pred)

                    if ( output=="wst"):                   # predict the wgt standardized with mean 0 and stand dev 1
                        wgt_train_pred = wgt_train
                        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
                        stan_sc.fit(wgt_train_pred)
                        wgt_train_pred = stan_sc.transform(wgt_train_pred)
                        wgt_val_pred = wgt_val
                        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
                        wgt_val_pred = stan_sc.transform(wgt_val_pred)

                    if (output=="lno"):                    # predict the absolute value of the logarithm of the wgt normalized
                        wgt_train_pred = np.abs(np.log(wgt_train))
                        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
                        norm_sc.fit(wgt_train_pred)
                        wgt_train_pred = norm_sc.transform(wgt_train_pred)
                        wgt_val_pred = np.abs(np.log(wgt_val))
                        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
                        wgt_val_pred = norm_sc.transform(wgt_val_pred)

                    if (output=="lst"):                    # predict the absolute value of the logarithm of the wgt standardized
                        wgt_train_pred = np.abs(np.log(wgt_train))
                        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
                        stan_sc.fit(wgt_train_pred)
                        wgt_train_pred = stan_sc.transform(wgt_train_pred)
                        wgt_val_pred = np.abs(np.log(wgt_val))
                        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
                        wgt_val_pred = stan_sc.transform(wgt_val_pred)

                    if (output=="wgt"):                    # predict the wgt
                        wgt_train_pred, wgt_val_pred = wgt_train, wgt_val

                    if (output=="lnw"):                    # predict the wgt
                        wgt_train_pred = np.abs(np.log(wgt_train)) 
                        wgt_val_pred =  np.abs(np.log(wgt_val))

                    time_2 = time.time()
                    time_init = time_2 - time_1



                    ##################################################
                    # prediction of weights
                    ##################################################

                    # define the model
                    tf.random.set_seed(seed_all) 
                    if (o_a == "linear"):
                        out_act_func = tf.keras.activations.linear
                    if (o_a == "leaky_relu"):
                        out_act_func = tf.keras.layers.LeakyReLU(alpha=0.1)

                    if (lays=="4x64"):
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(64, activation='relu', input_shape = (len(X_train[0]), )),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(1, activation=out_act_func)
                        ])
                    if (lays=="4x128"):
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(128, activation='relu', input_shape = (len(X_train[0]), )),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(1, activation=out_act_func)
                            ])
                    if (lays=="8x64"):
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(64, activation='relu', input_shape = (len(X_train[0]), )),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(1, activation=out_act_func)
                            ])
                    if (lays=="128to16"):
                        model = tf.keras.Sequential([
                            tf.keras.layers.Dense(128, activation='relu', input_shape = (len(X_train[0]), )),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(32, activation='relu'),
                            tf.keras.layers.Dense(16, activation='relu'),
                            tf.keras.layers.Dense(1, activation=out_act_func)
                            ])

                    tf.keras.backend.set_floatx("float64")


                    # loss function and compile the model 
                    if (lossfunc=="mse"):
                        loss = 'mean_squared_error'
                    if (lossfunc=="hub"):
                        loss = tf.keras.losses.Huber(delta=delta_hub)
                    if (lossfunc=="lch"):
                        loss = 'logcosh'


                    opt = tf.keras.optimizers.Adam(learning_rate=learn_rate)
                    model.compile(optimizer=opt, loss=loss) 

                    # training
                    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
                    history = model.fit(X_train, wgt_train_pred, validation_data=(X_val, wgt_val_pred), batch_size=batch_size, epochs=n_epochs, callbacks=[callback]) 

                    time_train = time.time() - time_2



                    ##################################################
                    # definitions of maxima functions
                    ##################################################

                    def max_quantile_reduction(array_s, r_ow, array_x=[]):
                        # define a reduced maximum such that the overweights' remaining contribution to the total sum of weights is lower or equal to r*total sum
                        part_sum = 0 
                        if (len(array_x) == 0):                      # max for s_i
                            max_s = 0
                            if (r_ow <= 0):                             # to test overwgt maxima
                                max_s = max(array_s) * (1-r_ow)
                                return max_s
                            else:
                                arr_s = np.sort(array_s)             # sorted s_i to determine s_max
                                tot_sum = np.sum(arr_s)
                                for i in range(len(arr_s)):
                                    part_sum += arr_s[i]
                                    if (part_sum >= tot_sum*(1-r_ow)):
                                        max_s = np.abs(arr_s[i])
                                        break
                                return max_s        
                        else:                                        # max for x_i
                            max_x = 0                                # for x_i the total sum is given by the sum over x_i*s_i
                            if (r_ow <= 0):
                                max_x = max(array_x) * (1-r_ow)
                                return max_x
                            else:
                                arr_s = [s for _,s in sorted(zip(array_x, array_s))]         # sort s_i respect to x_i for the second unwgt
                                arr_x = np.sort(array_x)
                                tot_sum = np.sum(arr_s*arr_x) 
                                for i in range(len(arr_x)):
                                    part_sum += arr_s[i]*arr_x[i]
                                    if (part_sum >= tot_sum*(1-r)):
                                        max_x = np.abs(arr_x[i])
                                        break
                                return max_x



                    ##################################################
                    # unweighting
                    ##################################################

                    time_3 = time.time()

                    s1_pred = model.predict(tf.convert_to_tensor(X_val))
                    s1_pred = np.reshape(s1_pred, len(s1_pred))
                    if (output=="lno"):
                        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
                        s1 = norm_sc.inverse_transform(s1) 
                        s1 = s1.reshape(len(s1))
                        s1 = np.e**(-s1) 
                    if (output=="lst"):
                        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
                        s1 = stan_sc.inverse_transform(s1) 
                        s1 = s1.reshape(len(s1))
                        s1 = np.e**(-s1) 
                    if (output=="wno"):
                        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
                        s1 = norm_sc.inverse_transform(s1) 
                        s1 = s1.reshape(len(s1))
                    if (output=="wst"):
                        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
                        s1 = stan_sc.inverse_transform(s1) 
                        s1 = s1.reshape(len(s1))
                    if(output=="wgt"):
                        s1 = s1_pred 
                    if(output=="lnw"):
                        s1 = np.e**(-s1) 
                    s1 = np.double(s1)


                    time_4 = time.time()
                    time_pred = time_4 - time_3

                    time_unwgt1 = 0
                    time_unwgt2 = 0


                    np.random.seed(seed_all)                     # each test has the same seed
                    r1 = 0.1                              # parameter of the maxima function for the first unwgt

                    time_5 = time.time()

                    # first unweighting
                    rand1 = np.random.rand(len(s1))              # random numbers for the first unwgt
                    w2 = np.empty(len(s1))                             # real wgt evaluated after first unwgt
                    s2 = np.empty(len(s1))                             # predicted wgt kept by first unwgt
                    z2 = np.empty(len(s1))                             # predicted wgt after first unwgt
                    x2 = np.empty(len(s1))                             # ratio between real and predicted wgt of kept events

                    s_max = max_quantile_reduction(s1, r_ow=r1)
                    j = 0
                    for i in range(len(s1)):                 # first unweighting, based on the predicted wgt
                        if (np.abs(s1[i])/s_max > rand1[i]):
                            s2[j] = s1[i]
                            z2[j] = np.sign(s1[i])*np.maximum(1, np.abs(s1[i])/s_max)      # kept event's wgt after first unwgt 
                            w2[j] = wgt_val[i] 
                            x2[j] = wgt_val[i]/np.abs(s1[i])
                            j += 1
                    s2 = s2[0:j]
                    z2 = z2[0:j]
                    w2 = w2[0:j]
                    x2 = x2[0:j]
                    eff1 = efficiency(s1, s_max)

                    time_6 = time.time()
                    time_unwgt1 += (time_6 - time_5)


                    # second unweighting
                    rand2 = np.random.rand(len(x2))
                    r2 = 0.02                          # parameter of the maxima function for the second unwgt

                    time_7 = time.time()

                    s3 = np.empty(len(s2))                         # predicted wgt kept by second unwgt
                    z3 = np.empty(len(s2))                         # predicted wgt after second unwgt
                    x3 = np.empty(len(s2))
                    x_max = max_quantile_reduction(x2*np.abs(z2), r_ow=r2)              # changed x_max definition as s_max
                    j = 0
                    for i in range(len(s2)):                 # second unweighting
                        if ((np.abs(z2[i])*x2[i]/x_max) > rand2[i]):
                            s3[j] = s2[i]
                            z3[j] = np.sign(z2[i])*np.maximum(1, np.abs(z2[i])*x2[i]/x_max)
                            x3[j] = x2[i]
                            j += 1 
                    s3 = s3[0:j] 
                    z3 = z3[0:j] 
                    x3 = z3[0:j]
                    eff2 = efficiency(x2, x_max)
                    feff = effective_gain(z3, eff1, eff2)
                    kish = f_kish(z3)

                    time_8 = time.time()
                    time_unwgt2 += (time_8 - time_7)

                    
                    if (count_table==0):
                        fig, ax = plt.subplots(figsize=(8.27, 11.69)) 
                        ax.set_axis_off() 
                        data_table = [ ["" for i in range(len(col_lab))] for j in range(row_max)]
                        for i in range(len(col_lab)):
                            data_table[0][i] = col_lab[i]
                        count_table += 1

                    data_table[count_table][0] = "layers: {} \nlearn rate: {} \nbatch size: {} \noutput activ: {}".format(lays, l_r, b_s, o_a)
                    data_table[count_table][1] = "{}".format(history.history['val_loss'][-1])
                    data_table[count_table][2] = "{}".format(len(history.history['val_loss']))
                    data_table[count_table][3] = "{} \n{}".format(eff1, eff2)
                    data_table[count_table][4] = "{}".format(kish)
                    data_table[count_table][5] = "{}".format(feff)
                    data_table[count_table][6] = "{} \n{} \n{}".format(time_pred, time_unwgt1, time_unwgt2)

                    count_table += 1

                    if (count_table==row_max):
                        table = ax.table(cellText = data_table, cellLoc ='center', loc ='upper left' )     
                        for k in range(row_max):
                            for r in range(0, len(col_lab)):
                                cell = table[k, r]
                                cell.set_height(0.05)
                        
                        pdf.savefig(fig)
                        plt.close(fig)                    
                        count_table = 0

    if (count_table<row_max and count_table>0):
                            table = ax.table(cellText = data_table, cellLoc ='center', loc ='upper left' )     
                            for k in range(row_max):
                                for r in range(0, len(col_lab)):
                                    cell = table[k, r]
                                    cell.set_height(0.05)
                            
                            pdf.savefig(fig)
                            plt.close(fig)                    
                            count_table = 0
