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


dataset = "7iter"            # 7iter, 8t10to6
load_module = False
cluster = False

if (cluster==True):
    path_scratch = "/globalscratch/ucl/cp3/lbeccati"
if (cluster==False):
    path_scratch = "/home/lb_linux/nn_unwgt"



##################################################
# reading
##################################################

time_0 = time.time()

# reading the momenta and weight of events
if (dataset=="8t10to6"):
    data = np.empty((8*10**6, 33)) 
if (dataset=="7iter"):
    data = np.empty((64000, 33)) 
with open("{}/info_wgt_events_{}.txt".format(path_scratch, dataset), 'r') as infof:
    print("Start readind events")
    event = 0
    # in each line of info: (px, py, pz, E) of ga, gb; e-, e+, g1, g2, d, dbar and wgt 
    for line in infof.readlines():
        data[event, :] = [float(i) for i in line.split()]
        if (event%(8*10**4)==0):
            print("Events read: {}% in {:.2f}s".format((event/(8*10**4)), time.time()-time_0))
        event +=1
#    if (output=="cno"):
#        data = [data[i] for i in range(len(data)) if data[i, -1]>10**(-9)]
#        data = np.reshape(data, (len(data), len(data[0])))
#val_len = len(data)//(ratio_train_val+1)
#train_len = (len(data)//(ratio_train_val+1)) * ratio_train_val + len(data)%(ratio_train_val+1)
val_len = len(data)//4
train_len = (len(data)//(4)) * 3 + len(data)%(4)

time_read = time.time() - time_0 



##################################################
# settings
##################################################

tests = [111111]

for test in tests:
    # test = ABCDEFG
    # A (layers)
    if (test//10**5==1):
        layers = "4x64" 
    if (test//10**5==2):
        layers = "4x128" 
    if (test//10**5==3):
        layers = "8x64" 
    # B (loss function)
    if ((test%10**5)//10**4==1):
        lossfunc = "mse" 
    if ((test%10**5)//10**4==2):
        lossfunc = "hub" 
    if ((test%10**5)//10**4==3):
        lossfunc = "lch" 
    if ((test%10**5)//10**4==4):
        lossfunc = "chi" 
    #C (inputs of the nn and their normalization)
    if ((test%10**4)//1000==1):
        input = "ptitf" 
    if ((test%10**4)//1000==2):
        input = "ptetf" 
    if ((test%10**4)//1000==3):
        input = "p3pap" 
    # D (output of the nn and its normalization)
    if ((test%1000)//100==1):
        output = "wno" 
    if ((test%1000)//100==2):
        output = "wst" 
    if ((test%1000)//100==3):
        output = "lno" 
    if ((test%1000)//100==4):
        output = "lst" 
    if ((test%1000)//100==5):
        output = "wgt" 
    if ((test%1000)//100==6):
        output = "lnw" 
    # E (unweighting method)
    if ((test%100)//10==1):
        unwgt = "new"
    if ((test%100)//10==2):
        unwgt = "pap"
    # F (max function)
    if (test%10==1):
        maxfunc = "mqr"
    if (test%10==2):
        maxfunc = "mmr"


    if (output=="wno" or output=="lno"):
        delta_hub = 0.2
    if (output=="wst" or output=="lst"):
        delta_hub = 3

    seed_all = 2 

    n_epochs = 5 
    learn_rate = 0.001
    ratio_train_val = 3                              # number of training events over number of validation events

    eff1_st = 0.683                                  # standard effeciencies for the first unwgt
    eff2_st = 0.020                                  # standard effeciencies for the second unwgt
    E_cm_pro = 13000                                 # energy of cm of protons
    t_ratio = 0.002       # t_surrogate / t_standard = [1/20, 1/50, 1/100, 1/500], t: time to compute one event

    part_name = ["e-", "e+", "g1", "g2", "d", "d~"]
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


    if (input=="cmsca"):
        E_cm_int = energy_cm(data[:, 0:8])
        beta_int = beta(data[:, 0:8])
        X_data = np.empty(shape=(len(data), 16))
        X_data[:, 0] = E_cm_int / E_cm_pro 
        X_data[:, 1] = beta_int
        for i in range(5):       # for e-, e+, g3, g4, d
            X_data[:, i*3+2] = data[:, i*4+8]                      # px_cm 
            X_data[:, i*3+3] = data[:, i*4+9]                      # py_cm 
            X_data[:, i*3+4] = -(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11] + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10] / E_cm_int      #pz_cm
        X_train, X_val = X_data[:train_len, :], X_data[-val_len:, :]
        Xt_no = X_train[:, 2:]
        Xt_no = np.reshape(Xt_no, (len(Xt_no)*15, 1))
        norm_sc.fit(Xt_no)                                  # find min max of X_train
        Xt_no = norm_sc.transform(Xt_no)                    # scale X_train between 0,1
        Xt_no = np.reshape(Xt_no, (len(Xt_no)//15, 15))
        Xv_no = X_val[:, 2:]
        Xv_no = np.reshape(Xv_no, (len(Xv_no)*15, 1))
        Xv_no = norm_sc.transform(Xv_no)                        # transform X_val with the same tranof of X_train
        Xv_no = np.reshape(Xv_no, (len(Xv_no)//15, 15))    

    if (input=="ppctf"):         # momentum normalized, cos theta, phi
        input_name = ["E_int/E_pro", "beta", "|p|/E_pro", "cos(theta)", "phi"]
        E_cm_int = energy_cm(data[:, 0:8])
        beta_int = beta(data[:, 0:8])
        X_data = np.empty(shape=(len(data), 16))
        X_data[:, 0] = E_cm_int / E_cm_pro 
        X_data[:, 1] = beta_int
        for i in range(5):       # for e-, e+, g3, g4, d
            X_data[:, i*3+2] = np.sqrt(data[:, i*4+8]**2 + data[:, i*4+9]**2 + (-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11]
            + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10]**2)) / E_cm_pro          # momentum modulus in the cm over E_int
            X_data[:, i*3+3] = ((-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11] + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10]) 
            / np.sqrt(data[:, i*4+9]**2 + data[:, i*4+8]**2 + (-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11]
            + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10])**2))                        # cos(theta angle) in parallel plane
            X_data[:, i*3+4] = np.arccos(data[:, i*4+8] / np.sqrt(data[:, i*4+8]**2+data[:, i*4+9]**2)) * np.sign(data[:, i*4+9])          # phi angle in transverse plane
        X_train, X_val = X_data[:train_len, :], X_data[-val_len:, :]

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
    """
        Xt_no, Xv_no = np.empty(len(X_train)), np.empty(len(X_val))            # one array to normalize all same inputs toghether
        for j in range(2):                 # to normalize E_int/E_pro and beta_cm
            Xt_no[:] = X_train[:, j]
            Xv_no[:] = X_val[:, j]
            Xt_no = np.reshape(Xt_no, (len(Xt_no), 1))
            Xv_no = np.reshape(Xv_no, (len(Xv_no), 1))
            norm_sc.fit(Xt_no)                                  # find min max of X_train
            Xt_no = norm_sc.transform(Xt_no)                    # scale X_train between 0,1
            Xv_no = norm_sc.transform(Xv_no) 
            Xt_no = np.reshape(Xt_no, len(Xt_no))
            Xv_no = np.reshape(Xv_no, len(Xv_no))
            X_train[:, j] = Xt_no[:]
            X_val[:, j] = Xv_no[:]
        Xt_no, Xv_no = np.empty(len(X_train)*5), np.empty(len(X_val)*5)            # one array to normalize all momenta toghether
        for j in range(3):                 # to normalize with J=0: p; j=1: theta; j=2: phi
            for i in range(5):
                Xt_no[i*train_len:(i+1)*train_len] = X_train[:, 2+j+i*3]
                Xv_no[i*val_len:(i+1)*val_len] = X_val[:, 2+j+i*3]
            Xt_no = np.reshape(Xt_no, (len(Xt_no), 1))
            Xv_no = np.reshape(Xv_no, (len(Xv_no), 1))
            norm_sc.fit(Xt_no)                                  # find min max of X_train
            Xt_no = norm_sc.transform(Xt_no)                    # scale X_train between 0,1
            Xv_no = norm_sc.transform(Xv_no) 
            Xt_no = np.reshape(Xt_no, len(Xt_no))
            Xv_no = np.reshape(Xv_no, len(Xv_no))
            for i in range(5):
                X_train[:, 2+j+i*3] = Xt_no[i*train_len:(i+1)*train_len]
                X_val[:, 2+j+i*3] = Xv_no[i*val_len:(i+1)*val_len]
    """

    if (input=="ptptf"):         # momentum transverse over E_pro, phi, theta
        input_name = ["E_int/E_pro", "beta", "p_T/E_pro", "theta", "phi"]
        E_cm_int = energy_cm(data[:, 0:8])
        beta_int = beta(data[:, 0:8])
        X_data = np.empty(shape=(len(data), 16))
        X_data[:, 0] = E_cm_int / E_cm_pro 
        X_data[:, 1] = beta_int
        for i in range(5):       # for e-, e+, g3, g4, d
            X_data[:, i*3+2] = np.sqrt(data[:, i*4+8]**2 + data[:, i*4+9]**2) / E_cm_pro          # transverse momentum  in the cm over E_pro
            X_data[:, i*3+3] = np.arccos((-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11] + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10])
            / np.sqrt(data[:, i*4+9]**2 + data[:, i*4+8]**2 + (-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11]
            + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10])**2))                        # theta angle in parallel plane  (0, pi)
            X_data[:, i*3+4] = np.arccos(data[:, i*4+8] / np.sqrt(data[:, i*4+8]**2+data[:, i*4+9]**2)) * np.sign(data[:, i*4+9])          # phi angle in transverse plane  (-pi, pi)
        X_train, X_val = X_data[:train_len, :], X_data[-val_len:, :]


    if (input=="yptno"):         # rapidity, momentum transverse, theta
        input_name = ["s_int", "beta", "y", "p_T", "theta"]
        E_cm_int = energy_cm(data[:, 0:8])
        beta_int = beta(data[:, 0:8])
        X_data = np.empty(shape=(len(data), 16))
        X_data[:, 0] = E_cm_int / E_cm_pro 
        X_data[:, 1] = beta_int
        for i in range(5):       # for e-, e+, g3, g4, d
            X_data[:, 2+i*3] = rapidity(data[:, 8+i*4:12+i*4])                 # rapidity in the lab
            X_data[:, 3+i*3] = np.sqrt(data[:, 8+i*4]**2 + data[:, 9+i*4]**2)      # transverse momentum
            X_data[:, 4+i*3] = np.arccos(data[:, i*4+8] / np.sqrt(data[:, i*4+8]**2+data[:, i*4+9]**2)) * np.sign(data[:, i*4+9])    # phi angle in transverse plane
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
    """
    def norm_fun(wgt, norm_wgt):
        if (norm_wgt=="wno"):
            min_no = 10**(-30)
            max_no = 10**(-4)
            res = (wgt-min_no)/(max_no-min_no)
        if (norm_wgt=="lno"):
            min_no = 8
            max_no = 68
            res = (-np.log(np.abs(wgt))-min_no)/(max_no-min_no)
        if (norm_wgt=="wst"):
            mean_st = 1.12*(10**(-7))
            std_st = 2.63*(10**(-7))
            res = (wgt - mean_st)/std_st
        if (norm_wgt=="lst"):
            mean_st = 16.65
            std_st = 1.29
            res = (-np.log(np.abs(wgt)) - mean_st)/std_st
        return res
    """

    wgt_train, wgt_val = np.abs(data[:train_len, -1]), np.abs(data[-val_len:, -1])      # take the abs of wgt to avoid problems
    wgt_train_pred, wgt_val_pred = np.empty(len(wgt_train)), np.empty(len(wgt_val))     # value to predict
    
    if (output=="wno" or output=="cno"):             # predict the wgt (or the wgt with a lower cut) normalized between 0 and 1
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
    if (layers=="4x64"):
        if (output=="wno" or output=="cno" or output=="lno"):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape = (len(X_train[0]), )),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape = (len(X_train[0]), )),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
    if (layers=="4x128"):
        if (output=="wno" or output=="cno" or output=="lno"):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape = (len(X_train[0]), )),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape = (len(X_train[0]), )),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
    if (layers=="8x64"):
        if (output=="wno" or output=="cno" or output=="lno"):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape = (len(X_train[0]), )),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape = (len(X_train[0]), )),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

    tf.keras.backend.set_floatx("float64")


    # loss function and compile the model 
    def chi_sqaure(w_true, w_pred):
        chisq = 0
        chisq = (w_pred - w_true)**2 / tf.math.abs(w_true)
        return chisq

    def mse_wgt(w_true, w_pred):
        w_len, w_size = w_true.get_shape()
        res = tf.multiply(tf.square(tf.subtract(w_true, w_pred)), tf.math.abs(w_true))/ w_len
        return res

    if (lossfunc=="mse"):
        loss = 'mean_squared_error'
    if (lossfunc=="hub"):
        loss = tf.keras.losses.Huber(delta=delta_hub)
    if (lossfunc=="lch"):
        loss = 'logcosh'
    if (lossfunc=="chi"):
        loss = chi_sqaure
    if (lossfunc=="wms"):
        loss = mse_wgt


    if (load_module==True):
        model = tf.keras.models.load_model("{}/model_nn_unwgt_{}_{}_{}_{}_{}_{}_seed{}_{}".format(path_scratch, layers, input, output, lossfunc, maxfunc, unwgt, seed_all, dataset))

    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        model.compile(optimizer=opt, loss=loss) 

        # training
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        history = model.fit(X_train, wgt_train_pred, validation_data=(X_val, wgt_val_pred), batch_size=1000, epochs=n_epochs, callbacks=[callback]) 

        time_train = time.time() - time_2

        # save model
        model.save("{}/model_nn_unwgt_{}_{}_{}_{}_{}_{}_seed{}_{}".format(path_scratch, layers, input, output, lossfunc, maxfunc, unwgt, seed_all, dataset))


    if (load_module==False):
        fig_tr, axs_tr = plt.subplots(figsize=(8.27, 6))
        axs_tr.plot(history.history['loss'], label="Training")
        axs_tr.plot(history.history['val_loss'], label="Validation")
        axs_tr.set_yscale('log')
        axs_tr.set(xlabel="epochs", ylabel="loss")
        axs_tr.legend(loc=0)
        fig_tr.savefig("{}/train_{}_{}_{}_{}_{}_{}_seed{}_{}.pdf".format(path_scratch, layers, input, output, lossfunc, maxfunc, unwgt, seed_all, dataset))



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
        arr_r = [0.2, 0.1, 0.05, 0.01] 
        #arr_r = [0.7, 0.5, 0.3, 0.1] 

    if (maxfunc=="mmr"):
        my_max = max_median_reduction
        arr_r = [100, 50, 10, 5]

    if (unwgt=="new"):
        arr_smax = np.empty(len(arr_r))
    if (unwgt=="pap"):
        arr_wmax = np.empty(len(arr_r))
    arr_eff1 = np.empty(len(arr_r))



    ##################################################
    # preparation of plots
    ##################################################
    
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, len(arr_r))) 

    bins_tr1 = np.linspace(0, 1, 50)
    bins_tr2 = np.linspace(-0.5, 0.5, 50)
    bins_beta = np.linspace(-1, 1, 50)
    bins_theta = np.linspace(0, np.pi, 50)
    bins_phi = np.linspace(-np.pi, np.pi, 50)
    bins_mee = np.linspace(0, 500, 50)

    bins_ws = np.logspace(np.log10(10**(-10)), np.log10(10**8), 70)
    bins_w = np.logspace(np.log10(10**(-10)), np.log10(10**(-4)), 50)
    if (output=="wno" or output=="cno"):
        bins_w_pred = np.logspace(-15, 0, 50)
    if (output=="lno"):
        bins_w_pred = np.logspace(-5, 0, 50)
    if (output=="wst" or output=="lst"):
        bins_w_pred = np.linspace(-10, 10, 50)
    if (output=="wgt"):
        bins_w_pred = bins_w
    if (output=="lnw"):
        bins_w_pred = np.logspace(5, 80, 50)

    plot_legend = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: 1000 | Learn. rate: {} \nEv_train: {} | Ev_val: {}
Input norm: {} | Output norm: {} \nLoss: {} | Max func: {} \nUnwgt method: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
        n_epochs, learn_rate, len(X_train), len(X_val), input, output, lossfunc, maxfunc, unwgt, seed_all)
    
    fig_ws, axs_ws = plt.subplots(2, figsize=(8.27, 11.69))
    fig_ws.legend(title = plot_legend)

    fig_zk1, axs_zk1 = plt.subplots(4, figsize=(8.27, 11.69))
    fig_zk1.legend(title = plot_legend)
    fig_zk2, axs_zk2 = plt.subplots(4, figsize=(8.27, 11.69))
    fig_zk2.legend(title = plot_legend)
    fig_zk3, axs_zk3 = plt.subplots(4, figsize=(8.27, 11.69))
    fig_zk3.legend(title = plot_legend)
    fig_zk4, axs_zk4 = plt.subplots(4, figsize=(8.27, 11.69))
    fig_zk4.legend(title = plot_legend)



    ##################################################
    # unweighting
    ##################################################

    mtx_xmax, mtx_eff2, mtx_kish, mtx_feff = np.empty((len(arr_r), len(arr_r))), np.empty((len(arr_r), len(arr_r))), np.empty((len(arr_r), len(arr_r))), np.empty((len(arr_r), len(arr_r)))

    time_3 = time.time()

    s1_pred = model.predict(X_val)
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
    if (output=="wno" or output=="cno"):
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

    time_4 = time.time()
    time_pred = time_4 - time_3

    time_unwgt1 = 0
    time_unwgt2 = 0

    for i_r1 in range(len(arr_r)):                   # loop to test different maxima conditions

        np.random.seed(seed_all)                     # each test has the same seed
        r = arr_r[i_r1]                              # parameter of the maxima function for the first unwgt

        time_5 = time.time()

        # first unweighting
        rand1 = np.random.rand(len(s1))              # random numbers for the first unwgt
        w2 = np.empty(len(s1))                             # real wgt evaluated after first unwgt
        s2 = np.empty(len(s1))                             # predicted wgt kept by first unwgt
        z2 = np.empty(len(s1))                             # predicted wgt after first unwgt
        x2 = np.empty(len(s1))                             # ratio between real and predicted wgt of kept events

        if (unwgt=="new"):                           # new method for the unweighting
            s_max = my_max(s1)
            arr_smax[i_r1] = s_max
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
            arr_eff1[i_r1] = efficiency(z2, s1, s_max)
        if (unwgt=="pap"):                           # paper method for the unwgt
            w_max = my_max(wgt_val)                  # unwgt done respect w_max
            arr_wmax[i_r1] = w_max
            j = 0
            for i in range(len(s1)):                 # first unwgt, based on the predicted wgt
                if (np.abs(s1[i])/w_max > rand1[i]):
                    s2[j] = s1[i] 
                    z2[j] = np.sign(s1[i])*np.maximum(1, np.abs(s1[i])/w_max)
                    w2[j] = wgt_val[i]    
                    x2[j] = wgt_val[i]/np.abs(s1[i])
                    j+=1
            s2 = s2[0:j]
            z2 = z2[0:j]
            w2 = w2[0:j]
            x2 = x2[0:j]
            arr_eff1[i_r1] = efficiency(z2, s1, w_max)

        time_6 = time.time()
        time_unwgt1 += (time_6 - time_5)

        axs_ws[1].hist(x=x2*np.abs(z2), bins=bins_ws, label="r1={}%".format(arr_r[i_r1]*100), color=colors[i_r1], histtype='step', lw=2, alpha=0.7)

        for i_r2 in range(len(arr_r)):               # to test all combinations of s_max and x_max 
            # second unweighting
            rand2 = np.random.rand(len(x2))
            r = arr_r[i_r2]                          # parameter of the maxima function for the second unwgt

            time_7 = time.time()

            s3 = np.empty(len(s2))                         # predicted wgt kept by second unwgt
            z3 = np.empty(len(s2))                         # predicted wgt after second unwgt
            x3 = np.empty(len(s2))
            z3_0ow = np.empty(len(s2))                     # final events with no overwgt
            z3_1ow = np.empty(len(s2))                     # final events with overwgt only in the first unwgt (reabsorbed)
            z3_2ow = np.empty(len(s2))                     # final events with overwgt only in the second unwgt 
            z3_12ow = np.empty(len(s2))                    # final events with overwgt in both unwgt
            if (unwgt=="new"):
                if (maxfunc=="mqr"):
    # # # # #                x_max = my_max(s2, x2*np.abs(z2))
                    x_max = my_max(x2*np.abs(z2))              # changed x_max definition as s_max
                if (maxfunc=="mmr"):
                    x_max = my_max(x2*np.abs(z2)) 
                j, j0, j1, j2, j12 = 0, 0, 0, 0, 0
                for i in range(len(s2)):                 # second unweighting
                    if ((np.abs(z2[i])*x2[i]/x_max) > rand2[i]):
                        s3[j] = s2[i]
                        z3[j] = np.sign(z2[i])*np.maximum(1, np.abs(z2[i])*x2[i]/x_max)
                        x3[j] = x2[i]
                        if (z2[i]==1 and z3[j]==1):
                            z3_0ow[j0] = 1
                            j0 += 1
                        if (z2[i]>1 and z3[j]==1):
                            z3_1ow[j1] = 1
                            j1 += 1
                        if (z2[i]==1 and z3[j]>1):
                            z3_2ow[j2] = z3[j]
                            j2 +=1
                        if (z2[i]>1 and z3[j]>1):
                            z3_12ow[j12] = z3[j]
                            j12 += 1
                        j += 1 
                s3 = s3[0:j] 
                z3 = z3[0:j] 
                x3 = z3[0:j]
                z3_0ow = z3_0ow[0:j0]
                z3_1ow = z3_1ow[0:j1] 
                z3_2ow = z3_2ow[0:j2] 
                z3_12ow = z3[0:j12]
                mtx_eff2[i_r1, i_r2] = efficiency(z3, x2, x_max)
                mtx_feff[i_r1, i_r2] = effective_gain(z3, arr_eff1[i_r1], mtx_eff2[i_r1, i_r2])

                time_8 = time.time()
                time_unwgt2 += (time_8 - time_7)

                # for zk plot only 
                mtx_kish[i_r1, i_r2] = f_kish(z3)
                if (len(z3)>=1):
                    if (i_r1==0):
                        axs_zk1[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk1[i_r2].set_ylim([1, 10**5])
                    axs_zk1[i_r2].set_yscale('log')
                    axs_zk1[i_r2].legend(loc='best')
                    axs_zk1[i_r2].set(xlabel="z3", ylabel="dN/dz3")
                    if (i_r1==1):
                        axs_zk2[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk2[i_r2].set_ylim([1, 10**5])
                    axs_zk2[i_r2].set_yscale('log')
                    axs_zk2[i_r2].legend(loc='best')
                    axs_zk2[i_r2].set(xlabel="z3", ylabel="dN/dz3")
                    if (i_r1==2):
                        axs_zk3[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk3[i_r2].set_ylim([1, 10**5])
                    axs_zk3[i_r2].set_yscale('log')
                    axs_zk3[i_r2].legend(loc='best')
                    axs_zk3[i_r2].set(xlabel="z3", ylabel="dN/dz3")
                    if (i_r1==3):
                        axs_zk4[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk4[i_r2].set_ylim([1, 10**5])
                    axs_zk4[i_r2].set_yscale('log')
                    axs_zk4[i_r2].legend(loc='best')
                    axs_zk4[i_r2].set(xlabel="z3", ylabel="dN/dz3")


            if (unwgt=="pap"):
                ztot = np.empty(len(s2))
    
                if (maxfunc=="mqr"):
    # # # # #                x_max = my_max(s2, x2)
                    x_max = my_max(s2, x2) 
                if (maxfunc=="mmr"):
                    x_max = my_max(x2) 
                j, j0, j1, j2, j12 = 0, 0, 0, 0, 0
                for i in range(len(s2)):                 # second unweighting
                    if ((x2[i]/x_max) > rand2[i]):
                        s3[j] = s2[i]
                        z3[j] = np.maximum(1, x2[i]/x_max)
                        ztot[j] = z2[i]*z3[j]
                        x3[j] = x2[i] 
                        if (z2[i]==1 and z3[j]==1):
                            z3_0ow[j0] = 1
                            j0 += 1
                        if (z2[i]>1 and z3[j]==1):
                            z3_1ow[j1] = 1
                            j1 += 1
                        if (z2[i]==1 and z3[j]>1):
                            z3_2ow[j2] = z3[j]
                            j2 +=1
                        if (z2[i]>1 and z3[j]>1):
                            z3_12ow[j12] = z3[j]
                            j12 += 1
                        j += 1
                s3 = s3[0:j] 
                x3 = z3[0:j]
                z3 = z3[0:j]
                ztot = ztot[0:j] 
                z3_0ow = z3_0ow[0:j0]
                z3_1ow = z3_1ow[0:j1] 
                z3_2ow = z3_2ow[0:j2] 
                z3_12ow = z3_12ow[0:j12]

                mtx_eff2[i_r1, i_r2] = efficiency(z3, x2, x_max) 
                mtx_feff[i_r1, i_r2] = effective_gain(ztot, arr_eff1[i_r1], mtx_eff2[i_r1, i_r2])

                time_8 = time.time()
                time_unwgt2 += (time_8 - time_7)

                # for zk plot only 
                mtx_kish[i_r1, i_r2] = f_kish(ztot) 
                if (len(z3)>=1):
                    if (i_r1==0):
                        axs_zk1[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk1[i_r2].set_ylim([1, 10**5])
                    axs_zk1[i_r2].set_yscale('log')
                    axs_zk1[i_r2].legend(loc='best')
                    axs_zk1[i_r2].set(xlabel="z3", ylabel="dN/dz3")
                    if (i_r1==1):
                        axs_zk2[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk2[i_r2].set_ylim([1, 10**5])
                    axs_zk2[i_r2].set_yscale('log')
                    axs_zk2[i_r2].legend(loc='best')
                    axs_zk2[i_r2].set(xlabel="z3", ylabel="dN/dz3")
                    if (i_r1==2):
                        axs_zk3[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk3[i_r2].set_ylim([1, 10**5])
                    axs_zk3[i_r2].set_yscale('log')
                    axs_zk3[i_r2].legend(loc='best')
                    axs_zk3[i_r2].set(xlabel="z3", ylabel="dN/dz3")
                    if (i_r1==3):
                        axs_zk4[i_r2].hist((z3_0ow, z3_1ow, z3_2ow, z3_12ow), bins=np.linspace(0.995, max(z3)+0.05, 15), 
                            color=['blue', 'yellow', 'orange', 'red'], label="""s_max {}% \nx_max{}% \nf_kish: {:.3f} \nN_0ow: {} \nN_1ow: {} \nN_2ow: {} \nN_12ow: {}""".format(
                                arr_r[i_r1]*100, arr_r[i_r2]*100, mtx_kish[i_r1, i_r2], len(z3_0ow), len(z3_1ow), len(z3_2ow), len(z3_12ow)))
                    axs_zk4[i_r2].set_ylim([1, 10**5])
                    axs_zk4[i_r2].set_yscale('log')
                    axs_zk4[i_r2].legend(loc='best')
                    axs_zk4[i_r2].set(xlabel="z3", ylabel="dN/dz3")

            mtx_xmax[i_r1, i_r2] = x_max
            #axs_eff[1].annotate(arr_r[i_r2], (x_max, mtx_eff2[i_r1, i_r2]))



    ##################################################
    # unweighting of standard sample with fixed kish factor
    ##################################################

    def effective_gain_st(eff1, eff2, eff_st):       # effective gain factor where the standard method computes all the matrix elements
        t_ratio = 0.002
        res = 1 / (t_ratio*eff_st/(eff1*eff2) + eff_st/eff2)
        return res
    """
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
        if (unwgt=="new"):
            axs_eff[3].plot(mtx_kish[i_r1], mtx_feff_st[i_r1], marker='.', color=colors[i_r1], label="s_max = {:.2e}".format(arr_smax[i_r1])) 
        if (unwgt=="pap"):
            axs_eff[3].plot(mtx_kish[i_r1], mtx_feff_st[i_r1], marker='.', color=colors[i_r1], label="w_max = {:.2e}".format(arr_wmax[i_r1])) 
    axs_eff[3].set(xlabel="Kish factor", ylabel="f_eff_st")
    axs_eff[3].legend
    axs_eff[3].legend(loc='best')
    """

    time_unwgt1 = time_unwgt1 / len(arr_r)
    time_unwgt2 = time_unwgt1 / (len(arr_r)**2)


    ##################################################
    # plot of results
    ##################################################

    if (input=="ptitf"):
        m_ee_train = np.sqrt( ( np.sqrt(X_train[:, 2]**2 * (1 + 1/(np.tan(X_train[:, 3]))**2)) + np.sqrt(X_train[:, 5]**2*(1 + 1/(np.tan(X_train[:, 6]))**2)) )**2 -
        ( (X_train[:, 2]/np.tan(X_train[:, 3]) + X_train[:, 5]/np.tan(X_train[:, 6]))**2 + (X_train[:, 2]*np.cos(X_train[:, 4]) + X_train[:, 5]*np.cos(X_train[:, 7]))**2 +
        (X_train[:, 2]*np.sin(X_train[:, 4]) + X_train[:, 5]*np.sin(X_train[:, 7]))**2 ) )      # invariant mass of e-e+
        
        m_ee_val = np.sqrt( ( np.sqrt(X_val[:, 2]**2 * (1 + 1/(np.tan(X_val[:, 3]))**2)) + np.sqrt(X_val[:, 5]**2*(1 + 1/(np.tan(X_val[:, 6]))**2)) )**2 -
        ( (X_val[:, 2]/np.tan(X_val[:, 3]) + X_val[:, 5]/np.tan(X_val[:, 6]))**2 + (X_val[:, 2]*np.cos(X_val[:, 4]) + X_val[:, 5]*np.cos(X_val[:, 7]))**2 +
        (X_val[:, 2]*np.sin(X_val[:, 4]) + X_val[:, 5]*np.sin(X_val[:, 7]))**2 ) )      # invariant mass of e-e+
        
        m_ee_train = m_ee_train * X_train[:, 0] * E_cm_pro      # normalize m_e-e+ respect s_pro
        m_ee_val = m_ee_val * X_val[:, 0] * E_cm_pro

    if (input=="p3pap"):
        m_ee_train = np.sqrt( (np.sqrt(X_train[:, 2]**2 + X_train[:, 3]**2 + X_train[:, 4]**2) + np.sqrt(X_train[:, 5]**2 + X_train[:, 6]**2 + X_train[:, 7]**2))**2 - 
        ( (X_train[:, 2] + X_train[:, 5])**2 + (X_train[:, 3] + X_train[:, 6])**2 + (X_train[:, 4] + X_train[:, 7])**2 ) )

        m_ee_val = np.sqrt( (np.sqrt(X_val[:, 2]**2 + X_val[:, 3]**2 + X_val[:, 4]**2) + np.sqrt(X_val[:, 5]**2 + X_val[:, 6]**2 + X_val[:, 7]**2))**2 - 
        ( (X_val[:, 2] + X_val[:, 5])**2 + (X_val[:, 3] + X_val[:, 6])**2 + (X_val[:, 4] + X_val[:, 7])**2 ) )
        
        m_ee_train = m_ee_train * E_cm_pro / 2     # normalize m_e-e+ respect s_pro
        m_ee_val = m_ee_val * E_cm_pro / 2

    #with PdfPages("{}/plot_{}_{}_{}_{}_{}_{}_seed{}_{}.pdf".format(path_scratch, layers, input, output, lossfunc, maxfunc, unwgt, seed_all, dataset)) as pdf: 
    with PdfPages("/home/lb_linux/nn_unwgt/plot_.pdf") as pdf: 
        # plot train and input distribution
            
        if (input=="ptitf"):
            for i_pag in range(6):

                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: 1000 | Learn. rate: {} \nEv_train: {} | Ev_val: {} 
Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, learn_rate, len(X_train), len(X_val), input, output, lossfunc, seed_all), loc=9)
            
                if (i_pag==0):
                    axs[0].set_yscale('log')
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel="dN/d({})".format(input_name[0]))
                    axs[0].hist(x=X_train[:, 0], bins=bins_tr1, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[0]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, 0], bins=bins_tr1, weights=wgt_val, label="{}_val".format(input_name[0]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, 0], bins=bins_tr1, weights=s1, label="{}_pred".format(input_name[0]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel="dN/d({})".format(input_name[1]))
                    axs[1].hist(x=X_train[:, 1], bins=bins_beta, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[1]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, 1], bins=bins_beta, weights=wgt_val, label="{}_val".format(input_name[1]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, 1], bins=bins_beta, weights=s1, label="{}_pred".format(input_name[1]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="m_e-e+ [GeV]", ylabel="dN/d(m_e-e+)")
                    axs[2].hist(x=m_ee_train, bins=bins_mee, weights=wgt_train/ratio_train_val, label="m_e-e+_train", color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_val, bins=bins_mee, weights=wgt_val, label="m_e-e+_val", color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_val, bins=bins_mee, weights=s1, label="m_e-e+_pred", color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[2]))
                    axs[0].hist(x=X_train[:, i_pag*3-1], bins=bins_tr1, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[2]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, i_pag*3-1], bins=bins_tr1, weights=wgt_val, label="{}_val".format(input_name[2]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, i_pag*3-1], bins=bins_tr1, weights=s1, label="{}_pred".format(input_name[2]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[3]))
                    axs[1].hist(x=X_train[:, i_pag*3], bins=bins_theta, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[3]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, i_pag*3], bins=bins_theta, weights=wgt_val, label="{}_val".format(input_name[3]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, i_pag*3], bins=bins_theta, weights=s1, label="{}_pred".format(input_name[3]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].legend(loc='best')

                    if (i_pag<5):
                        axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[4]))
                        axs[2].hist(x=X_train[:, i_pag*3+1], bins=bins_phi, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[4]), color='purple', histtype='step', lw=2, alpha=0.7)
                        axs[2].hist(x=X_val[:, i_pag*3+1], bins=bins_phi, weights=wgt_val, label="{}_val".format(input_name[4]), color='teal', histtype='step', lw=2, alpha=0.7)
                        axs[2].hist(x=X_val[:, i_pag*3+1], bins=bins_phi, weights=s1, label="{}_pred".format(input_name[4]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                        axs[2].legend(loc='best')
                    else:
                        axs[2].remove()
                
                pdf.savefig(fig)

        if (input=="p3pap"):
            for i_pag in range(7):
                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: 1000 | Learn. rate: {} \nEv_train: {} | Ev_val: {} 
Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, learn_rate, len(X_train), len(X_val), input, output, lossfunc, seed_all), loc=9)
                if (i_pag==0):
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel="dN/d({})".format(input_name[0]))
                    axs[0].hist(x=X_train[:, 0], bins=bins_beta, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[0]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, 0], bins=bins_beta, weights=wgt_val, label="{}_val".format(input_name[0]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, 0], bins=bins_beta, weights=s1, label="{}_pred".format(input_name[0]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel="dN/d({})".format(input_name[1]))
                    axs[1].hist(x=X_train[:, 1], bins=bins_beta, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[1]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, 1], bins=bins_beta, weights=wgt_val, label="{}_val".format(input_name[1]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, 1], bins=bins_beta, weights=s1, label="{}_pred".format(input_name[1]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="m_e-e+ [GeV]", ylabel="dN/d(m_e-e+)")
                    axs[2].hist(x=m_ee_train, bins=bins_mee, weights=wgt_train/ratio_train_val, label="m_e-e+_train", color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_val, bins=bins_mee, weights=wgt_val, label="m_e-e+_val", color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_val, bins=bins_mee, weights=s1, label="m_e-e+_pred", color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[2]))
                    axs[0].hist(x=X_train[:, i_pag*3-1], bins=bins_tr2, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[2]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, i_pag*3-1], bins=bins_tr2, weights=wgt_val, label="{}_val".format(input_name[2]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_val[:, i_pag*3-1], bins=bins_tr2, weights=s1, label="{}_pred".format(input_name[2]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[3]))
                    axs[1].hist(x=X_train[:, i_pag*3], bins=bins_tr2, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[3]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, i_pag*3], bins=bins_tr2, weights=wgt_val, label="{}_val".format(input_name[3]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_val[:, i_pag*3], bins=bins_tr2, weights=s1, label="{}_pred".format(input_name[3]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[4]))
                    axs[2].hist(x=X_train[:, i_pag*3+1], bins=bins_tr2, weights=wgt_train/ratio_train_val, label="{}_train".format(input_name[4]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=X_val[:, i_pag*3+1], bins=bins_tr2, weights=wgt_val, label="{}_val".format(input_name[4]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=X_val[:, i_pag*3+1], bins=bins_tr2, weights=s1, label="{}_pred".format(input_name[4]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                pdf.savefig(fig)


        # plot wgt mean respect inputs
        if (input=="ptitf"):
            for i_pag in range(6):

                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: 1000 | Learn. rate: {} \nEv_train: {} | Ev_val: {} 
    Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, learn_rate, len(X_train), len(X_val), input, output, lossfunc, seed_all), loc=9)

                if (i_pag==0):
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel="mean wgt({})".format(input_name[0]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 0]>=bins_tr1[i] and X_train[j, 0]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, 0]>=bins_tr1[i] and X_val[j, 0]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, 0]>=bins_tr1[i] and X_val[j, 0]<bins_tr1[i+1]]) for i in range(49)]
                    axs[0].plot(bins_tr1[1:], w_mean_tr, label="{}_train".format(input_name[0]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_va, label="{}_val".format(input_name[0]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_pr, label="{}_pred".format(input_name[0]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel="dN/d({})".format(input_name[1]))
                    w_tr, w_va, w_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 1]>=bins_beta[i] and X_train[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, 1]>=bins_beta[i] and X_val[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, 1]>=bins_beta[i] and X_val[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_beta[1:], w_mean_tr, label="{}_train".format(input_name[1]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_va, label="{}_val".format(input_name[1]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_pr, label="{}_pred".format(input_name[1]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="m_e-e+ [GeV]", ylabel="dN/d(m_e-e+)")
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if m_ee_train[j]>=bins_mee[i] and m_ee_train[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if m_ee_val[j]>=bins_mee[i] and m_ee_val[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if m_ee_val[j]>=bins_mee[i] and m_ee_val[j]<bins_mee[i+1]]) for i in range(49)]
                    axs[2].plot(bins_mee[1:], w_mean_tr, label="{}_train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_va, label="{}_val".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_pr, label="{}_pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[2]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3-1]>=bins_tr1[i] and X_train[j, i_pag*3-1]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, i_pag*3-1]>=bins_tr1[i] and X_val[j, i_pag*3-1]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, i_pag*3-1]>=bins_tr1[i] and X_val[j, i_pag*3-1]<bins_tr1[i+1]]) for i in range(49)]
                    axs[0].plot(bins_tr1[1:], w_mean_tr, label="{}_train".format(input_name[2]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_va, label="{}_val".format(input_name[2]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_pr, label="{}_pred".format(input_name[2]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[3]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3]>=bins_theta[i] and X_train[j, i_pag*3]<bins_theta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, i_pag*3]>=bins_theta[i] and X_val[j, i_pag*3]<bins_theta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, i_pag*3]>=bins_theta[i] and X_val[j, i_pag*3-1]<bins_theta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_theta[1:], w_mean_tr, label="{}_train".format(input_name[3]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_theta[1:], w_mean_va, label="{}_val".format(input_name[3]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_theta[1:], w_mean_pr, label="{}_pred".format(input_name[3]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].legend(loc='best')

                    if (i_pag<5):
                        axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[4]))
                        w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                        w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3+1]>=bins_phi[i] and X_train[j, i_pag*3+1]<bins_phi[i+1]]) for i in range(49)]
                        w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, i_pag*3+1]>=bins_phi[i] and X_val[j, i_pag*3+1]<bins_phi[i+1]]) for i in range(49)]
                        w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, i_pag*3+1]>=bins_phi[i] and X_val[j, i_pag*3+1]<bins_phi[i+1]]) for i in range(49)]
                        axs[2].plot(bins_phi[1:], w_mean_tr, label="{}_train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                        axs[2].plot(bins_phi[1:], w_mean_va, label="{}_val".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                        axs[2].plot(bins_phi[1:], w_mean_pr, label="{}_pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                        axs[2].legend(loc='best')
                    else:
                        axs[2].remove()

                pdf.savefig(fig)

        if (input=="p3pap"):
            for i_pag in range(7):

                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: 1000 | Learn. rate: {} \nEv_train: {} | Ev_val: {} 
    Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, learn_rate, len(X_train), len(X_val), input, output, lossfunc, seed_all), loc=9)

                if (i_pag==0):
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel="mean wgt({})".format(input_name[0]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 0]>=bins_beta[i] and X_train[j, 0]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, 0]>=bins_beta[i] and X_val[j, 0]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, 0]>=bins_beta[i] and X_val[j, 0]<bins_beta[i+1]]) for i in range(49)]
                    axs[0].plot(bins_beta[1:], w_mean_tr, label="{}_train".format(input_name[0]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_va, label="{}_val".format(input_name[0]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_pr, label="{}_pred".format(input_name[0]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel="dN/d({})".format(input_name[1]))
                    w_tr, w_va, w_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 1]>=bins_beta[i] and X_train[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, 1]>=bins_beta[i] and X_val[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, 1]>=bins_beta[i] and X_val[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_beta[1:], w_mean_tr, label="{}_train".format(input_name[1]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_va, label="{}_val".format(input_name[1]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_pr, label="{}_pred".format(input_name[1]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="m_e-e+ [GeV]", ylabel="dN/d(m_e-e+)")
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if m_ee_train[j]>=bins_mee[i] and m_ee_train[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if m_ee_val[j]>=bins_mee[i] and m_ee_val[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if m_ee_val[j]>=bins_mee[i] and m_ee_val[j]<bins_mee[i+1]]) for i in range(49)]
                    axs[2].plot(bins_mee[1:], w_mean_tr, label="{}_train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_va, label="{}_val".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_pr, label="{}_pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[2]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3-1]>=bins_beta[i] and X_train[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, i_pag*3-1]>=bins_beta[i] and X_val[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, i_pag*3-1]>=bins_beta[i] and X_val[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    axs[0].plot(bins_beta[1:], w_mean_tr, label="{}_train".format(input_name[2]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_va, label="{}_val".format(input_name[2]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_pr, label="{}_pred".format(input_name[2]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[3]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3]>=bins_beta[i] and X_train[j, i_pag*3]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, i_pag*3]>=bins_beta[i] and X_val[j, i_pag*3]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, i_pag*3]>=bins_beta[i] and X_val[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_beta[1:], w_mean_tr, label="{}_train".format(input_name[3]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_va, label="{}_val".format(input_name[3]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_pr, label="{}_pred".format(input_name[3]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel="dN/d({})".format(input_name[4]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3+1]>=bins_beta[i] and X_train[j, i_pag*3+1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_val[j] for j in range(len(wgt_val)) if X_val[j, i_pag*3+1]>=bins_beta[i] and X_val[j, i_pag*3+1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_val[j, i_pag*3+1]>=bins_beta[i] and X_val[j, i_pag*3+1]<bins_beta[i+1]]) for i in range(49)]
                    axs[2].plot(bins_beta[1:], w_mean_tr, label="{}_train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                    axs[2].plot(bins_beta[1:], w_mean_va, label="{}_val".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                    axs[2].plot(bins_beta[1:], w_mean_pr, label="{}_pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                    axs[2].legend(loc='best')

                pdf.savefig(fig)

        fig, axs = plt.subplots(2, figsize=(8.27, 11.69)) 

        # plot ws
        fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: 1000 | Learn. rate: {} \nEv_train: {} | Ev_val: {} 
Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {} \nTime read : {:.3f}s | Time init : {:.3f}s 
Time pred : {:.3f}s | Time unwgt1 : {:.3f}s | Time unwgt2 : {:.3f}s""".format(len(X_train[0]), layers,
                n_epochs, learn_rate, len(X_train), len(X_val), input, output, lossfunc, seed_all, time_read, time_init, time_pred, time_unwgt1, time_unwgt2))

        x1 = np.divide(wgt_val, s1)
        if (output=="wgt"):
            axs_ws[0].set_xscale('log')

        # plot predicted output
        axs[0].set(xlabel="output normalized", ylabel="dN/dw")
        axs[0].hist(x=wgt_train_pred, bins=bins_w_pred, label="w_train", weights=(1/ratio_train_val)*np.ones_like(wgt_train_pred), color='teal', histtype='step', lw=3, alpha=0.5)
        axs[0].hist(x=wgt_val_pred, bins=bins_w_pred, label="w_val", color='darkblue', histtype='step', lw=3, alpha=0.5)
        axs[0].hist(x=s1_pred, bins=bins_w_pred, label="s_pred", color='purple', histtype='step', lw=3, alpha=0.5)
        axs[0].legend(loc='best')
        if (output=="wno" or output=="lno" or output=="cno"):
            axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        
        axs[1].set(xlabel="w", ylabel="w/s")
        h2 = axs[1].hist2d(wgt_val, x1, bins=[bins_w, bins_ws])
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        plt.colorbar(h2[3], ax=axs[1]) 

        pdf.savefig(fig)

        axs_ws[0].set(xlabel="w/s", ylabel="dN/d(w/s)")
        axs_ws[0].hist(x1, bins=bins_ws)
        axs_ws[0].axvline(x=1, color='black', ls='--')
        axs_ws[0].set_xscale('log')
        axs_ws[0].set_yscale('log')

        axs_ws[1].legend(loc='best')
        axs_ws[1].set(xlabel="z2*w2/s2", ylabel="dN/d(z2*w2/s2)")
        axs_ws[1].axvline(x=1, color='black', ls='--')
        axs_ws[1].set_xscale('log')
        axs_ws[1].set_yscale('log')

        pdf.savefig(fig_ws)


        # plot eff
        fig, axs = plt.subplots(2, figsize=(8.27, 11.69)) 

        if (unwgt=="new"):
            axs[0].set(xlabel="s_max", ylabel="eff_1")
            axs[0].plot(arr_smax, arr_eff1, marker='.', markersize=15)
        if (unwgt=="pap"):
            axs[0].set(xlabel="w_max", ylabel="eff_1")
            axs[0].plot(arr_wmax, arr_eff1, marker='.', markersize=15)
        axs[0].legend(loc='best')
        axs[1].set(xlabel="x_max", ylabel="eff_2")
        """
        for i_r1 in range(len(arr_r)):                   # to mark the values for max=real_max with larger points
            #axs_eff[0].annotate(arr_r[i_r1], xy=(arr_smax[i_r1], arr_eff1[i_r1]))
            if (arr_r[i_r1]==0):
                if (unwgt=="new"):
                    axs[0].scatter(arr_smax[i_r1], arr_eff1[i_r1], s=50)
                if (unwgt=="pap"):
                    axs[0].scatter(arr_wmax[i_r1], arr_eff1[i_r1], s=50)
            for i_r2 in range(len(arr_r)):
                if (arr_r[i_r1]==0):
                    axs[1].scatter(mtx_xmax[i_r1, i_r2], mtx_eff2[i_r1, i_r2], s=50)
                else:
                    if (arr_r[i_r2]==0):
                        axs[1].scatter(mtx_xmax[i_r1, i_r2], mtx_eff2[i_r1, i_r2], s=50)
        """
        if (unwgt=="new"):
            for i in range(len(arr_r)):                      # plot the curve of the efficiencies in function of the x_max with fixed s_max
                axs[1].plot(mtx_xmax[i, :], mtx_eff2[i], marker='.', markersize=15, color=colors[i], label="s_max: {}%".format(arr_r[i]*100))
        if (unwgt=="pap"):
            for i in range(len(arr_r)):                      # plot the curve of the efficiencies in function of the x_max with fixed s_max
                axs[1].plot(mtx_xmax[i, :], mtx_eff2[i], marker='.', markersize=15, color=colors[i], label="s_max: {}%".format(arr_r[i]*100))
        axs[1].legend(loc='best')
        pdf.savefig(fig)


        fig, axs = plt.subplots(2, figsize=(8.27, 11.69)) 

        for i in range(len(arr_r)):                      # x, y labels of the f_eff colormap
            if (unwgt=="new"):
                axs[0].text(-0.5, 0.3+i, "s_max: {}%".format(arr_r[i]*100), fontsize = 8)
            if (unwgt=="pap"):
                axs[0].text(-0.5, 0.3+i, "s_max: {}%".format(arr_r[i]*100), fontsize = 8)
            axs[0].text(0.3+i, -0.5, "x_max {}%".format(arr_r[i]*100), fontsize = 8)
        #axs[0].axis('off')
        axs[0].set_title(label="f_eff")
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        h1 = axs[0].pcolormesh(mtx_feff, cmap=cmap)
        ticks_feff = np.linspace(np.min(mtx_feff), np.max(mtx_feff), 10, endpoint=True)
        plt.colorbar(h1, ax=axs[0], ticks=ticks_feff)

        # plot of plR (plot 1/R) 
        x_plR = np.linspace(1, len(arr_r)*len(arr_r), len(arr_r)*len(arr_r))
        y1_plR = [ 1 / ( arr_eff1[i//len(arr_r)] * ( mtx_kish[i//len(arr_r),i%len(arr_r)] * mtx_eff2[i//len(arr_r),i%len(arr_r)] / (eff1_st*eff2_st) - 1 ) ) for i in range(len(arr_r)*len(arr_r))] 
        y10_plR = [ 1 / ( arr_eff1[i//len(arr_r)] * ( mtx_kish[i//len(arr_r),i%len(arr_r)] * mtx_eff2[i//len(arr_r),i%len(arr_r)] / (eff1_st*eff2_st*10) - 1 ) ) for i in range(len(arr_r)*len(arr_r))] 
        # bar color with the kish factor
        kish_dif = np.max(mtx_kish) - np.min(mtx_kish) 
        c_plR = [cmap((mtx_kish[i//len(arr_r), i%len(arr_r)]-np.min(mtx_kish))/kish_dif) for i in range(len(arr_r)*len(arr_r))]
        norm_plR = mpl.colors.Normalize(vmin=np.min(mtx_kish), vmax=np.max(mtx_kish)) 
        sm_plR = plt.cm.ScalarMappable(cmap=cmap, norm=norm_plR)
        sm_plR.set_array([])
        axs[1].scatter(x_plR, y1_plR, color=c_plR, s=100, alpha=0.7, marker='o', label="f_eff>1")
        axs[1].scatter(x_plR, y10_plR, color=c_plR, s=100, alpha=0.7, marker='X', label="f_eff>10")
        x_plR_min, x_plR_max = axs[1].get_xlim()
        #axs[1].hlines(y=t_ratio, xmin=x_plR_min, xmax=x_plR_max, label="R \n(min value for positive gain)", color='green')
        plt.colorbar(sm_plR, ax=axs[1], ticks=np.linspace(np.min(mtx_kish), np.max(mtx_kish), 5))
        #axs[1].text(-1.5, -0.7 , "Overwgt unwgt 1: \nOverwgt unwgt 2:", fontsize = 6)
        #for i in range(len(arr_r)*len(arr_r)): 
        #    axs[1].text(0+i, -0.7 , "{}% \n{}%".format(arr_r[i//len(arr_r)]*100,arr_r[i%len(arr_r)]*100), fontsize = 6)
        #axs[1].set_xticklabels(["{}% \n{}%".format(arr_r[i//len(arr_r)]*100,arr_r[i%len(arr_r)]*100) for i in range(len(arr_r)*len(arr_r))])
        axs[1].set_xticks(x_plR)
        axs[1].set_xticklabels(["{}% \n{}%".format(arr_r[i//len(arr_r)]*100,arr_r[i%len(arr_r)]*100) for i in range(len(arr_r)*len(arr_r))])
        axs[1].tick_params(axis='x', which='major', labelsize=6)
        axs[1].set_ylim([10**(-3), 10**4])
        axs[1].set_ylabel("Overwgt unwgt 1, 2")
        axs[1].set_yscale('log')
        axs[1].set_ylabel("(t_MG/t_nn)_min")
        axs[1].legend(loc='best')
        pdf.savefig(fig)


        # plot zk
        pdf.savefig(fig_zk1)
        pdf.savefig(fig_zk2)
        pdf.savefig(fig_zk3)
        pdf.savefig(fig_zk4)


    print("\n--------------------------------------------------")
    print(plot_legend)
