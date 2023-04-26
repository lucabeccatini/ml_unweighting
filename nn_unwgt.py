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

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels


#where = "cluster"            # cluster or local
where = "local"              # cluster or local

#dataset = "8t10to6"          # 7iter, 10to6, 8t10to6, 7m80prt, 8m90prt, 8m95prt
dataset = "7m80prt"          # 7iter, 10to6, 8t10to6, 7m80prt, 8m90prt, 8m95prt
#dataset = "7iter"            # 7iter, 10to6, 8t10to6, 7m80prt, 8m90prt, 8m95prt

#load_module = True
load_module = False

#n_epochs = 1000 
#n_epochs = 100 
n_epochs = 5 

channel = "G128"             # G128 , G304 
#channel = "G304"             # G128 , G304 

#test_sample = "1m"           # 16k, 100k, 1m
test_sample = "16k"          # 16k, 100k, 1m


if (test_sample=="16k"):
    test_len = 16000
if (test_sample=="100k"):
    test_len = 10**5
if (test_sample=="1m"):
    test_len = 10**6
    
if (where=="cluster"):
    path_scratch = "/globalscratch/ucl/cp3/lbeccati"
if (where=="local"):
    path_scratch = "/home/lb_linux/nn_unwgt/{}".format(channel)



##################################################
# reading
##################################################

time_0 = time.time()

# reading the momenta and weight of events
if (dataset=="8t10to6"):
    n_data = 8*10**6
    train_len = 5*10**6
    val_len = 2*10**6
if (dataset=="10to6"):
    n_data = 10**6
    train_len = 7*10**5
    val_len = 2*10**5
if (dataset=="7iter"):
    n_data = 64000
    train_len = 32000
    val_len = 16000
if (dataset=="7m80prt"): 
    n_data = 2250000
    train_len = 900000
    val_len = 325000
if (dataset=="7m90prt"): 
    n_data = 1700000
    train_len = 500000
    val_len = 200000
if (dataset=="7m95prt"): 
    n_data = 1350000
    train_len = 250000
    val_len = 100000

ratio_train_test = train_len/test_len
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

# partially unwgt training samples use the original events for test (not the same event)
if (dataset=="7m80prt" or dataset=="7m90prt" or dataset=="7m95prt"):
    with open("{}/info_wgt_events_{}_8t10to6.txt".format(path_scratch, channel), 'r') as infof:
        event_test = 0
        for line in infof.readlines():
            if (event_test>=7*10**6):            # up to 7m used for unwgt training sample
                data[event, :] = [float(i) for i in line.split()]
                if (event%(n_data//100)==0):
                    print("Events read: {}% in {:.2f}s".format((event/(n_data//100)), time.time()-time_0))
                event+=1
            event_test+=1



##################################################
# settings
##################################################

tests = [2151811]
#tests = [2111511, 2111211, 2111611, 2111411, 2111711, 2111811] 
#tests = [2121513, 2121213, 2121613, 2121413, 2121713, 2121813] 
#tests = [2151511, 2151211, 2151611, 2151411, 2151711, 2151811] 


for test in tests:
    # test = ABCDEFGH
    # A (layers)
    if (test//10**6==1):
        layers = "4x64" 
    if (test//10**6==2):
        layers = "4x128" 
    if (test//10**6==3):
        layers = "8x64" 
    # B (loss function)
    if ((test%10**6)//10**5==1):
        lossfunc = "mse" 
    if ((test%10**6)//10**5==2):
        lossfunc = "hub" 
    if ((test%10**6)//10**5==3):
        lossfunc = "lch" 
    if ((test%10**6)//10**5==4):
        lossfunc = "chi" 
    #C (minimum on the prediction)
    if ((test%10**5)//10**4==1):
        minpred = "nmp"                # nmp: no minimum prediction
    if ((test%10**5)//10**4==2):
        minpred = "btq"                # btq: (s_low) before training quantile (reduction method)
    if ((test%10**5)//10**4==3):
        minpred = "btf"                # btf: (s_low) before training (smaller than a given) fraction 
    if ((test%10**5)//10**4==4):
        minpred = "btm"                # btm: (s_low) before training (respect w) max
    if ((test%10**5)//10**4==5):
        minpred = "atq"                # atq: (s_low) after training quantile (reduction method)
    if ((test%10**5)//10**4==6):
        minpred = "atf"                # atf: (s_low) after training (smaller than a given) fraction 
    if ((test%10**5)//10**4==7):
        minpred = "atm"                # atm: (s_low) after training (respect w) max
    #D (inputs of the nn and their normalization)
    if ((test%10**4)//10**3==1):
        input = "ptitf" 
    if ((test%10**4)//10**3==2):
        input = "ptptf" 
    if ((test%10**4)//10**3==3):
        input = "p3pap" 
    # E (output of the nn and its normalization)
    if ((test%10**3)//10**2==1):
        output = "wno"                 # w normalized
    if ((test%10**3)//10**2==2):
        output = "wst"                 # w standardized
    if ((test%10**3)//10**2==3):
        output = "lno"                 # ln(w) normalized
    if ((test%10**3)//10**2==4):
        output = "lst"                 # ln(w) standardized
    if ((test%10**3)//10**2==5):
        output = "wgt"                 # w 
    if ((test%10**3)//10**2==6):
        output = "lnw"                 # ln(w)
    if ((test%10**3)//10**2==7):
        output = "rww"                 # ln(1 + w/s_low)
    if ((test%10**3)//10**2==8):
        output = "rst"                 # ln(1 + w/s_low) standardized
    # G (unweighting method)
    if ((test%10**2)//10**1==1):
        unwgt = "new"                  # new unweighting method
    if ((test%10**2)//10**1==2):
        unwgt = "pap"                  # unweighting method of the SHERPA paper
    # F (output activation function)
    if (test%10==1):
        output_activation = "lin"                # linear
    if (test%10==2):
        output_activation = "lre"                # leaky relu
    if (test%10==3 or minpred=="btq" or minpred=="btf" or minpred=="btm"):
        output_activation = "rel"                # relu

    r_low = 0.999            # r_s to define s_low as minimum prediction

    if (output=="wno" or output=="lno"):
        delta_hub = 0.2
    if (output=="wst" or output=="lst"):
        delta_hub = 3

    seed_all = 2 
    maxfunc = "mqr"                    # mqr or mmr

    
    if (unwgt=="new"):
        batch_size = 500
        learn_rate = 0.0001
    if (unwgt=="pap"):
        batch_size = 1000
        learn_rate = 0.001

    if (channel=="G128"):
        eff1_st = 0.6832                                  # standard effeciencies for the first unwgt
        eff2_st = 0.0205                                  # standard effeciencies for the second unwgt
        t_st = 0.00248                                    # average time to unwgt one event for MG
        #t_ratio = 0.02                                   # t_surrogate / t_standard = 1/50, t: time to compute one event
    if (channel=="G304"):
        eff1_st = 0.6537                                  # standard effeciencies for the first unwgt
        eff2_st = 0.0455                                  # standard effeciencies for the second unwgt
        t_st = 0.00387                                    # average time to unwgt one event for MG
    E_cm_pro = 13000                                 # energy of cm of protons
    t_nn = 0.00003

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

    def effective_gain(z_f, eff1, eff2, t_su):                           # effective gain factor
        # eff gain given by the ratio between the total time T of the standard method and the surrogate one to obtain the same results
        res = f_kish(z_f) / ((t_su/t_st)*(eff1_st*eff2_st)/(eff1*eff2) + (eff1_st*eff2_st/eff2))
        return res



    ##################################################
    # definitions of maxima functions
    ##################################################

    def max_quantile_reduction(array_s, r_ow, array_x=[]):
        # define a reduced maximum such that the overweights' remaining contribution to the total sum of weights is lower or equal to r*total sum
        part_sum = 0 
        if (len(array_x) == 0):                      # max for s_i
            max_s = 0
            if (r_ow <= 0):                             # to test overweighted maxima
                max_s = max(array_s) * (1-r_ow)
                return max_s
            else:
                arr_s = np.sort(array_s)             # sorted s_i to determine s_max
                tot_sum = np.sum(arr_s)
                for i in range(len(arr_s)):
                    part_sum += arr_s[i]
                    if (part_sum >= tot_sum*(1-r_ow)):
                        max_s = arr_s[i]
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

    def max_median_reduction(arr, r_ow):
        # define a reduced maximum such that it is the median over the maxima of different unweighted samples
        if (r_ow <= 0):                                 # to test overwgt maxima
            max_s = max(arr) * (1-r_ow)
            return max_s
        else:
            n_max = 50                               # number of maxima used for the median
            arr = np.flip(np.sort(arr))              # reversed sorted input array
            arr_max = np.zeros(n_max)
            max_r = arr[0] * r_ow 
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
        arr_r1 = [0.1, 0.01, 0.001, 0.0001] 
        arr_r2 = [0.04, 0.03, 0.02, 0.01] 

    if (maxfunc=="mmr"):
        my_max = max_median_reduction
        arr_r1 = [100, 50, 10, 5]
        arr_r2 = [100, 50, 10, 5]

    if (unwgt=="new"):
        arr_smax = np.empty(len(arr_r1))
    if (unwgt=="pap"):
        arr_wmax = np.empty(len(arr_r1))
    arr_eff1 = np.empty(len(arr_r1))



    ##################################################
    # input normalization
    ##################################################

    def energy_cm(X):
        res = np.sqrt((X[:, 3]+X[:, 7])**2 - (X[:, 2]+X[:, 6])**2)       # E_cm = sqrt((E_g1+E_g2)**2-(pz_g1+pz_g2)**2)
        return res

    def beta(X):                                     # beta in the lab frame
        res = (X[:, 2]+X[:, 6]) / (X[:, 3]+X[:, 7])      # beta = (pz_g1+pz_g2)/(E_g1+E_g2))
        return res

    def rapidity(X):                                 # rapidity in the lab frame
        res = 0.5 * np.log((X[:, 3]+X[:, 2]) / (X[:, 3]-X[:, 2]))        # y = 1/2 * ln((E+pz)/(E-pz))
        return res


    #t_init_i = time.time()
    if (input=="ptitf"):         # momentum transverse over E_int, phi, theta
        input_name = [r"$\sqrt{\hat{S}}/\sqrt{S}$", r"$\beta$", r"$p_T/\sqrt{\hat{S}}$", r"$\theta$", r"$\phi$"]
        #input_name = ["E_int/E_pro", "beta", r"$p_T/E_int", "theta", "phi"]
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
        X_train, X_val, X_test = X_data[:train_len, :], X_data[train_len:(train_len+val_len), :], X_data[(train_len+val_len):(train_len+val_len+test_len), :]

    if (input=="p3pap"):
        input_name = [r"$p_z(g_a)$", r"$p_z(g_b)$", r"$p_x$", r"$p_y$", r"$p_z$"]
        X_data = np.empty(shape=(len(data), 20))
        X_data[:, 0] = data[:, 2]
        X_data[:, 1] = data[:, 6]
        for i in range(6):       # for e-, e+, g3, g4, d, d~
            X_data[:, i*3+2] = data[:, i*4+8]                      # px
            X_data[:, i*3+3] = data[:, i*4+9]                      # py 
            X_data[:, i*3+4] = data[:, i*4+10]
        X_data = X_data/(E_cm_pro/2)
        X_train, X_val, X_test = X_data[:train_len, :], X_data[train_len:(train_len+val_len), :], X_data[(train_len+val_len):(train_len+val_len+test_len), :]


    """
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

    if (input=="ptptf"):         # momentum transverse over E_pro, phi, theta
        input_name = ["E_int/E_pro", "beta", "p_T/E_pro", "theta", "phi"]
        E_cm_int = energy_cm(data[:, 0:8])
        beta_int = beta(data[:, 0:8])
        X_data = np.empty(shape=(len(data), 16))
        X_data[:, 0] = E_cm_int / E_cm_pro 
        X_data[:, 1] = beta_int
        phi_d = np.arccos(data[:, 24] / np.sqrt(data[:, 24]**2+data[:, 25]**2)) * np.sign(data[:, 25])           # phi angle of d 
        for i in range(5):       # for e-, e+, g3, g4, d
            X_data[:, i*3+2] = np.sqrt(data[:, i*4+8]**2 + data[:, i*4+9]**2) / E_cm_pro          # transverse momentum  in the cm over E_pro
            X_data[:, i*3+3] = np.arccos((-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11] + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10])
            / np.sqrt(data[:, i*4+9]**2 + data[:, i*4+8]**2 + (-(beta_int/np.sqrt(1-beta_int**2))*data[:, i*4+11]
            + (1/np.sqrt(1-beta_int**2))*data[:, i*4+10])**2))                        # theta angle in parallel plane  (0, pi)
            if (i<4):
                X_data[:, i*3+4] = np.arccos(data[:, i*4+8] / np.sqrt(data[:, i*4+8]**2+data[:, i*4+9]**2)) * np.sign(data[:, i*4+9]) - phi_d            # phi angle in transverse plane  (-pi, pi)
                X_data[:, i*3+4] -= (np.abs(X_data[:, i*3+4])//np.pi)*2*np.pi*np.sign(X_data[:, i*3+4])          # to rinormalize phi between (-pi, pi)
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
    """



    ##################################################
    # output inizialization
    ##################################################

    wgt_train, wgt_val, wgt_test = np.abs(data[:train_len, -1]), np.abs(data[train_len:(train_len+val_len), -1]), np.abs(data[(train_len+val_len):(train_len+val_len+test_len), -1])      # take the abs of wgt to avoid problems
    wgt_train_pred, wgt_val_pred, wgt_test_pred = np.empty(len(wgt_train)), np.empty(len(wgt_val)), np.empty(len(wgt_test))     # value to predict


    # evaluation of s_low
    s_low = 0
    if (minpred=="btq" or minpred=="atq"):
        s_low = my_max(wgt_train, r_ow=r_low)              # s_low = 1/1000 of the contribution of the cross section
    if (minpred=="btf" or minpred=="atf"):
        s_low = np.sort(wgt_train)[len(wgt_train)//100]    # s_low larger than 99% of events
    if (minpred=="btm" or minpred=="atm"): 
        s_low = my_max(wgt_train, r_ow=0.10) * 10**(-3)    # s_low = w_max / 10^6


    if (output=="lst"):                    # predict the absolute value of the logarithm of the wgt standardized
        wgt_train_pred = np.log(np.abs(wgt_train))
        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
        stan_sc.fit(wgt_train_pred)
        wgt_train_pred = stan_sc.transform(wgt_train_pred)
        wgt_val_pred = np.log(np.abs(wgt_val))
        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
        wgt_val_pred = stan_sc.transform(wgt_val_pred)
        wgt_test_pred = np.log(np.abs(wgt_test))
        wgt_test_pred = np.reshape(wgt_test_pred, (len(wgt_test_pred), 1))
        wgt_test_pred = stan_sc.transform(wgt_test_pred)

    if (output=="lnw"):                    # predict the wgt
        wgt_train_pred = np.abs(np.log(wgt_train)) 
        wgt_val_pred =  np.abs(np.log(wgt_val))
        wgt_test_pred =  np.abs(np.log(wgt_test))

    if ( output=="wst"):                   # predict the wgt standardized with mean 0 and stand dev 1
        wgt_train_pred = wgt_train
        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
        stan_sc.fit(wgt_train_pred)
        wgt_train_pred = stan_sc.transform(wgt_train_pred)
        wgt_val_pred = wgt_val
        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
        wgt_val_pred = stan_sc.transform(wgt_val_pred)
        wgt_test_pred = wgt_test
        wgt_test_pred = np.reshape(wgt_test_pred, (len(wgt_test_pred), 1))
        wgt_test_pred = stan_sc.transform(wgt_test_pred)

    if (output=="wgt"):                    # predict the wgt
        wgt_train_pred = wgt_train
        wgt_val_pred = wgt_val
        wgt_test_pred = wgt_test

    if (output=="rww"):                    # predict log( 1 + w/w_low)
        if (minpred=="nmp"):
            s_low = my_max(wgt_train, r_ow=r_low)              # s_low = 1/1000 of the contribution of the cross section
        wgt_train_pred = np.log(1 + wgt_train/s_low) 
        wgt_val_pred =  np.log(1 + wgt_val/s_low)
        wgt_test_pred =  np.log(1 + wgt_test/s_low)

    if (output=="rst"):                    # predict log( 1 + w/w_low) standardized
        if (minpred=="nmp"):
            s_low = my_max(wgt_train, r_ow=r_low)              # s_low = 1/1000 of the contribution of the cross section
        wgt_train_pred = np.log(1 + wgt_train/s_low) 
        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
        stan_sc.fit(wgt_train_pred)
        wgt_train_pred = stan_sc.transform(wgt_train_pred)
        wgt_val_pred =  np.log(1 + wgt_val/s_low)
        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
        wgt_val_pred = stan_sc.transform(wgt_val_pred)
        wgt_test_pred =  np.log(1 + wgt_test/s_low)
        wgt_test_pred = np.reshape(wgt_test_pred, (len(wgt_test_pred), 1))
        wgt_test_pred = stan_sc.transform(wgt_test_pred)


    if (minpred!="nmp"): 
        if (output=="lnw"):
            s_low_pred = np.log(s_low)
        if (output=="lst"):
            s_low_pred = np.log(s_low)
            s_low_pred = np.reshape(s_low_pred, (1, 1))
            s_low_pred = stan_sc.transform(s_low_pred)
            s_low_pred = s_low_pred[0, 0]
        if (output=="wgt"):
            s_low_pred = s_low
        if (output=="wst"):
            s_low_pred = s_low
            s_low_pred = np.reshape(s_low_pred, (1, 1))
            s_low_pred = stan_sc.transform(s_low_pred)
            s_low_pred = s_low_pred[0, 0]
        if (output=="rww"):
            s_low_pred = np.log(2)                # s_low' = ln(1 + s_low/s_low)
        if (output=="rst"):
            s_low_pred = np.log(2)
            s_low_pred = np.reshape(s_low_pred, (1, 1))
            s_low_pred = stan_sc.transform(s_low_pred)
            s_low_pred = s_low_pred[0, 0]

    """
    # add test to de-comment 
    if (output=="wno"):             # predict the wgt (or the wgt with a lower cut) normalized between 0 and 1
        wgt_train_pred = wgt_train
        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
        norm_sc.fit(wgt_train_pred)
        wgt_train_pred = norm_sc.transform(wgt_train_pred)
        wgt_val_pred = wgt_val
        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
        wgt_val_pred = norm_sc.transform(wgt_val_pred)
    
    
    if (output=="lno"):                    # predict the absolute value of the logarithm of the wgt normalized
        wgt_train_pred = np.log(np.abs(wgt_train))
        wgt_train_pred = np.reshape(wgt_train_pred, (len(wgt_train_pred), 1))
        norm_sc.fit(wgt_train_pred)
        wgt_train_pred = norm_sc.transform(wgt_train_pred)
        wgt_val_pred = np.log(np.abs(wgt_val))
        wgt_val_pred = np.reshape(wgt_val_pred, (len(wgt_val_pred), 1))
        wgt_val_pred = norm_sc.transform(wgt_val_pred)
    """



    ##################################################
    # prediction of weights
    ##################################################

    # define the model
    tf.random.set_seed(seed_all) 
    tf.keras.backend.set_floatx("float64")

    if (output_activation == "lin"):
        out_act_func = tf.keras.activations.linear
    if (output_activation == "lre"):
        out_act_func = tf.keras.layers.LeakyReLU(alpha=0.1)
    if (output_activation == "rel"):
        out_act_func = tf.keras.layers.ReLU()

    if (layers=="4x64"):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape = (len(X_train[0]), )),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation=out_act_func)
        ])
    if (layers=="4x128"):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape = (len(X_train[0]), )),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation=out_act_func)
            ])
    if (layers=="8x64"):
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
        
        
    # lambda layer to add s_low to the output of the nn
    def lambda_bias(y):
        res = y + s_low_pred
        return res

    lambda_layer = tf.keras.layers.Lambda(lambda_bias)

    if (minpred=="btq" or minpred=="btf" or minpred=="btm"):
        model.add(lambda_layer)


    # loss function and compile the model 
    def chi_sqaure(w_true, w_pred):
        chisq = 0
        chisq = (w_pred - w_true)**2 / tf.math.abs(w_true)
        return chisq

    if (lossfunc=="mse"):
        loss = 'mean_squared_error'
    if (lossfunc=="hub"):
        loss = tf.keras.losses.Huber(delta=delta_hub)
    if (lossfunc=="lch"):
        loss = 'logcosh'
    if (lossfunc=="chi"):
        loss = chi_sqaure

    if (load_module==True):
        model = tf.keras.models.load_model("{}/model_nn_unwgt_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed{}_{}".format(path_scratch, channel, layers, lossfunc, maxfunc, minpred, input, output, output_activation, unwgt, seed_all, dataset))

    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        model.compile(optimizer=opt, loss=loss) 

        # training
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        history = model.fit(X_train, wgt_train_pred, validation_data=(X_val, wgt_val_pred), batch_size=batch_size, epochs=n_epochs, callbacks=[callback]) 

        # save model
        model.save("{}/model_nn_unwgt_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed{}_{}".format(path_scratch, channel, layers, lossfunc, maxfunc, minpred, input, output, output_activation, unwgt, seed_all, dataset))


    if (load_module==False):
        fig_tr, axs_tr = plt.subplots(figsize=(8.27, 6))
        axs_tr.plot(history.history['loss'], label="Training")
        axs_tr.plot(history.history['val_loss'], label="Validation")
        axs_tr.set_yscale('log')
        axs_tr.set(xlabel="epochs", ylabel="loss")
        axs_tr.legend(loc=0)
        fig_tr.savefig("{}/train_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed{}_{}.pdf".format(path_scratch, channel, layers, lossfunc, maxfunc, minpred, input, output, output_activation, unwgt, seed_all, dataset))
        plt.close(fig_tr)



    ##################################################
    # preparation of plots
    ##################################################
    
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, len(arr_r1))) 

    if (input=="ptitf"):
        bins_tr1 = np.linspace(0, 1, 50)
    if (input=="ptptf" or input=="pepap"):
        bins_tr1 = np.linspace(0, 0.5, 50)
    bins_tr2 = np.linspace(-0.5, 0.5, 50)
    bins_beta = np.linspace(-1, 1, 50)
    bins_theta = np.linspace(0, np.pi, 50)
    bins_phi = np.linspace(-np.pi, np.pi, 50)
    bins_mee = np.linspace(0, 500, 50)

    bins_ws = np.logspace(np.log10(10**(-6)), np.log10(10**4), 70)
    bins_w = np.logspace(np.log10(10**(-15)), np.log10(10**(-3)), 70)
    if (output=="wno"): 
        bins_w_pred = np.logspace(-15, 0, 50)
    if (output=="lno"):
        bins_w_pred = np.logspace(-5, 0, 50)
    if (output=="wst" or output=="lst" or output=="rst"):
        bins_w_pred = np.linspace(-15, 15, 50)
    if (output=="wgt"):
        bins_w_pred = bins_w
    if (output=="lnw"):
        bins_w_pred = np.linspace(np.log(10**(-15)), np.log(10**(-3)), 50)
    if (output=="rww"):
        bins_w_pred = np.logspace(np.log(10**(-5)), np.log(25), 50)

    plot_legend = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: {} | Learn. rate: {} \nEv_train: {} | Ev_test: {}
Input norm: {} | Output norm: {} | Min pred: {} \nLoss: {} | Max func: {} \nUnwgt method: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
        n_epochs, batch_size , learn_rate, len(X_train), len(X_test), input, output, minpred, lossfunc, maxfunc, unwgt, seed_all)
    
    fig_ws, axs_ws = plt.subplots(2, figsize=(8.27, 11.69))
    fig_ws.legend(title = plot_legend)

    fig_tab, axs_tab = plt.subplots(1, figsize=(8.27, 11.69))  
    #col_lab = [r"$r_s$" "\n" r"$r_x$", r"$\varepsilon_1$" "\n" r"$\varepsilon_2$", r"$\alpha$", r"$f_{eff}$", "time init \ntime pred \ntime unwgt 1 \ntime unwgt 2", r"$\langle t_{nn}\rangle$"]
    col_lab = [r"$r_s$" "\n" r"$r_x$", r"$\varepsilon_1$" "\n" r"$\varepsilon_2$", r"$\alpha$", r"$f_{eff}$"]
    row_max = len(arr_r1)*len(arr_r2) + 1
    count_table = 1
    axs_tab.set_axis_off() 
    data_table = [ ["" for i in range(len(col_lab))] for j in range(row_max)]
    for i in range(len(col_lab)):
        data_table[0][i] = col_lab[i]

    fig_zk1, axs_zk1 = plt.subplots(4, figsize=(6, 11.69))
    fig_zk2, axs_zk2 = plt.subplots(4, figsize=(6, 11.69))
    fig_zk3, axs_zk3 = plt.subplots(4, figsize=(6, 11.69))
    fig_zk4, axs_zk4 = plt.subplots(4, figsize=(6, 11.69))



    ##################################################
    # unweighting
    ##################################################
    

    mtx_xmax, mtx_eff2, mtx_kish, mtx_feff = np.empty((len(arr_r1), len(arr_r2))), np.empty((len(arr_r1), len(arr_r2))), np.empty((len(arr_r1), len(arr_r2))), np.empty((len(arr_r1), len(arr_r2)))

    t_pred_i = time.time()

    s1_pred = model.predict(tf.convert_to_tensor(X_test))
    s1_pred = np.reshape(s1_pred, len(s1_pred))
    if (output=="lno"):
        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
        s1 = norm_sc.inverse_transform(s1) 
        s1 = s1.reshape(len(s1))
        s1 = np.exp(s1) 
    if (output=="lst"):
        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
        s1 = stan_sc.inverse_transform(s1) 
        s1 = s1.reshape(len(s1))
        s1 = np.exp(s1) 
    if (output=="wno"):
        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
        s1 = norm_sc.inverse_transform(s1) 
        s1 = s1.reshape(len(s1))
    if (output=="wst"):
        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
        s1 = stan_sc.inverse_transform(s1) 
        s1 = s1.reshape(len(s1))
    if (output=="wgt"):
        s1 = s1_pred 
    if (output=="lnw"):
        s1 = np.exp(s1_pred) 
    if (output=="rww"):
        s1 = (np.exp(s1_pred) - 1) * s_low 
    if (output=="rst"):
        s1 = np.reshape(s1_pred, (len(s1_pred), 1))
        s1 = stan_sc.inverse_transform(s1) 
        s1 = s1.reshape(len(s1))
        s1 = (np.exp(s1) - 1) * s_low 
    s1 = np.double(s1) 

    if (minpred=="atq" or minpred=="atf" or minpred=="atm"):
        s1 = np.maximum(s1, s_low)


    t_pred_f = time.time()
    time_pred = t_pred_f - t_pred_i


    for i_r1 in range(len(arr_r1)):                   # loop to test different maxima conditions

        np.random.seed(seed_all)                     # each test has the same seed
        r = arr_r1[i_r1]                              # parameter of the maxima function for the first unwgt

        # first unweighting
        rand1 = np.random.rand(len(s1))              # random numbers for the first unwgt
        w2 = np.empty(len(s1))                             # real wgt evaluated after first unwgt
        s2 = np.empty(len(s1))                             # predicted wgt kept by first unwgt
        z2 = np.empty(len(s1))                             # predicted wgt after first unwgt
        x2 = np.empty(len(s1))                             # ratio between real and predicted wgt of kept events

        #t_unwgt1_i = time.time()

        if (unwgt=="new"):                           # new method for the unweighting
            s_max = my_max(s1, r_ow=r)
            arr_smax[i_r1] = s_max
            j = 0
            for i in range(len(s1)):                 # first unweighting, based on the predicted wgt
                if (np.abs(s1[i])/s_max > rand1[i]):
                    s2[j] = s1[i]
                    z2[j] = np.sign(s1[i])*np.maximum(1, np.abs(s1[i])/s_max)      # kept event's wgt after first unwgt 
                    w2[j] = wgt_test[i] 
                    x2[j] = wgt_test[i]/np.abs(s1[i])
                    j += 1
            s2 = s2[0:j]
            z2 = z2[0:j]
            w2 = w2[0:j]
            x2 = x2[0:j]
            arr_eff1[i_r1] = efficiency(s1, s_max)
        if (unwgt=="pap"):                           # paper method for the unwgt
            w_max = my_max(wgt_test, r_ow=r)                  # unwgt done respect w_max
            arr_wmax[i_r1] = w_max
            j = 0
            for i in range(len(s1)):                 # first unwgt, based on the predicted wgt
                if (np.abs(s1[i])/w_max > rand1[i]):
                    s2[j] = s1[i] 
                    z2[j] = np.sign(s1[i])*np.maximum(1, np.abs(s1[i])/w_max)
                    w2[j] = wgt_test[i]    
                    x2[j] = wgt_test[i]/np.abs(s1[i])
                    j+=1
            s2 = s2[0:j]
            z2 = z2[0:j]
            w2 = w2[0:j]
            x2 = x2[0:j]
            arr_eff1[i_r1] = efficiency(s1, w_max)

        #t_unwgt1_f = time.time()
        #time_unwgt1 = t_unwgt1_f - t_unwgt1_i 


        for i_r2 in range(len(arr_r2)):               # to test all combinations of s_max and x_max 
            # second unweighting
            rand2 = np.random.rand(len(x2))
            r = arr_r2[i_r2]                          # parameter of the maxima function for the second unwgt

            s3 = np.empty(len(s2))                         # predicted wgt kept by second unwgt
            z3 = np.empty(len(s2))                         # predicted wgt after second unwgt
            x3 = np.empty(len(s2))
            z3_0ow = np.empty(len(s2))                     # final events with no overwgt
            z3_1ow = np.empty(len(s2))                     # final events with overwgt only in the first unwgt (reabsorbed)
            z3_2ow = np.empty(len(s2))                     # final events with overwgt only in the second unwgt 
            z3_12ow = np.empty(len(s2))                    # final events with overwgt in both unwgt
            
            #t_unwgt2_i = time.time()

            if (unwgt=="new"):
                if (maxfunc=="mqr"):
                    # # # # #x_max = my_max(s2, r, x2*np.abs(z2))
                    x_max = my_max(x2*np.abs(z2), r_ow=r)              # changed x_max definition as s_max
                if (maxfunc=="mmr"):
                    x_max = my_max(x2*np.abs(z2), r_ow=r) 
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
                z3_12ow = z3_12ow[0:j12]

            if (unwgt=="pap"):
                ztot = np.empty(len(s2))
                if (maxfunc=="mqr"):
                    # # # # #x_max = my_max(s2, x2)
                    x_max = my_max(array_s=s2, array_x=x2, r_ow=r) 
                if (maxfunc=="mmr"):
                    x_max = my_max(x2, r_ow=r) 
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
                        #if (z2[i]>1 and z3[j]==1):
                        #    z3_1ow[j1] = 1
                        #    j1 += 1
                        if (z2[i]==1 and z3[j]>1):
                            z3_2ow[j2] = z3[j]
                            j2 +=1
                        if (z2[i]>1):
                            z3_12ow[j12] = z3[j]
                            j12 += 1
                        j += 1
                s3 = s3[0:j] 
                x3 = x3[0:j]
                z3 = z3[0:j]
                ztot = ztot[0:j] 
                z3_0ow = z3_0ow[0:j0]
                z3_1ow = z3_1ow[0:0] 
                z3_2ow = z3_2ow[0:j2] 
                z3_12ow = z3_12ow[0:j12]

            #t_unwgt2_f = time.time()
            #time_unwgt2 = t_unwgt2_f - t_unwgt2_i

            #t_nn = (time_init + time_pred + time_unwgt1 + time_unwgt2)/len(wgt_test)

            mtx_xmax[i_r1, i_r2] = x_max
            mtx_eff2[i_r1, i_r2] = efficiency(x2, x_max)
            if (unwgt=="new"):
                mtx_kish[i_r1, i_r2] = f_kish(z3)
                mtx_feff[i_r1, i_r2] = effective_gain(z3, arr_eff1[i_r1], mtx_eff2[i_r1, i_r2], t_nn)
            if (unwgt=="pap"):
                mtx_kish[i_r1, i_r2] = f_kish(ztot) 
                mtx_feff[i_r1, i_r2] = effective_gain(ztot, arr_eff1[i_r1], mtx_eff2[i_r1, i_r2], t_nn)


            # table
            data_table[count_table][0] = "{}%\n{}%".format(arr_r1[i_r1]*100, arr_r2[i_r2]*100)
            data_table[count_table][1] = "{:.3f}\n{:.3f}".format(arr_eff1[i_r1], mtx_eff2[i_r1, i_r2])
            data_table[count_table][2] = "{:.3f}".format(mtx_kish[i_r1, i_r2])
            data_table[count_table][3] = "{:.3f}".format(mtx_feff[i_r1, i_r2])
            #data_table[count_table][4] = "{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}".format(time_init, time_pred, time_unwgt1, time_unwgt2)
            #data_table[count_table][5] = "{:.6f}".format(t_nn)
            count_table += 1


            # plot zk overweight
            if (len(z3)>=1):
                if (i_r1==0):
                    axs_zk1[i_r2].hist([z3_0ow, z3_1ow, z3_2ow, z3_12ow], bins=np.linspace(0.501, 10.501, 20), 
                        color=['blue', 'yellow', 'orange', 'red'], stacked=True, ec="k", 
                        label=[r"$N_0^{ow}: $"+"{}".format(len(z3_0ow)), r"$N_1^{ow}: $"+"{}".format(len(z3_1ow)), r"$N_2^{ow}: $"+"{}".format(len(z3_2ow)), r"$N_{12}^{ow}: $"+"{}".format(len(z3_12ow))])
                    axs_zk1[i_r2].set_ylim([1, 10**5])
                    axs_zk1[i_r2].set_yscale('log')
                    axs_zk1[i_r2].legend(title=r"$r_s: $"+"{}%\n".format(arr_r1[i_r1]*100)+r"$r_x: $"+"{}%\n".format(arr_r2[i_r2]*100)+r"$\alpha: $"+"{:.3f}".format(mtx_kish[i_r1, i_r2]), loc='best')
                    axs_zk1[i_r2].set(xlabel=r"$\tilde{w}$", ylabel=r"$\frac{dN}{d(\tilde{w})}$")
                    axs_zk1[i_r2].set_xticks(np.linspace(1, 10, 10))

                if (i_r1==1):
                    axs_zk2[i_r2].hist([z3_0ow, z3_1ow, z3_2ow, z3_12ow], bins=np.linspace(0.501, 10.501, 20), 
                        color=['blue', 'yellow', 'orange', 'red'], stacked=True, ec="k",  
                        label=[r"$N_0^{ow}: $"+"{}".format(len(z3_0ow)), r"$N_1^{ow}: $"+"{}".format(len(z3_1ow)), r"$N_2^{ow}: $"+"{}".format(len(z3_2ow)), r"$N_{12}^{ow}: $"+"{}".format(len(z3_12ow))])
                    axs_zk2[i_r2].set_ylim([1, 10**5])
                    axs_zk2[i_r2].set_yscale('log')
                    axs_zk2[i_r2].legend(title=r"$r_s: $"+"{}%\n".format(arr_r1[i_r1]*100)+r"$r_x: $"+"{}%\n".format(arr_r2[i_r2]*100)+r"$\alpha: $"+"{:.3f}".format(mtx_kish[i_r1, i_r2]), loc='best')
                    axs_zk2[i_r2].set(xlabel=r"$\tilde{w}$", ylabel=r"$\frac{dN}{d(\tilde{w})}$")
                    axs_zk2[i_r2].set_xticks(np.linspace(1, 10, 10))
                if (i_r1==2):
                    axs_zk3[i_r2].hist([z3_0ow, z3_1ow, z3_2ow, z3_12ow], bins=np.linspace(0.501, 10.501, 20), 
                        color=['blue', 'yellow', 'orange', 'red'], stacked=True, ec="k", 
                        label=[r"$N_0^{ow}: $"+"{}".format(len(z3_0ow)), r"$N_1^{ow}: $"+"{}".format(len(z3_1ow)), r"$N_2^{ow}: $"+"{}".format(len(z3_2ow)), r"$N_{12}^{ow}: $"+"{}".format(len(z3_12ow))])
                    axs_zk3[i_r2].set_ylim([1, 10**5])
                    axs_zk3[i_r2].set_yscale('log')
                    axs_zk3[i_r2].legend(title=r"$r_s: $"+"{}%\n".format(arr_r1[i_r1]*100)+r"$r_x: $"+"{}%\n".format(arr_r2[i_r2]*100)+r"$\alpha: $"+"{:.3f}".format(mtx_kish[i_r1, i_r2]), loc='best')
                    axs_zk3[i_r2].set(xlabel=r"$\tilde{w}$", ylabel=r"$\frac{dN}{d(\tilde{w})}$")
                    axs_zk3[i_r2].set_xticks(np.linspace(1, 10, 10))
                if (i_r1==3):
                    axs_zk4[i_r2].hist([z3_0ow, z3_1ow, z3_2ow, z3_12ow], bins=np.linspace(0.501, 10.501, 20), 
                        color=['blue', 'yellow', 'orange', 'red'], stacked=True, ec="k", 
                        label=[r"$N_0^{ow}: $"+"{}".format(len(z3_0ow)), r"$N_1^{ow}: $"+"{}".format(len(z3_1ow)), r"$N_2^{ow}: $"+"{}".format(len(z3_2ow)), r"$N_{12}^{ow}: $"+"{}".format(len(z3_12ow))])
                    axs_zk4[i_r2].set_ylim([1, 10**5])
                    axs_zk4[i_r2].set_yscale('log')
                    axs_zk4[i_r2].legend(title=r"$r_s: $"+"{}%\n".format(arr_r1[i_r1]*100)+r"$r_x: $"+"{}%\n".format(arr_r2[i_r2]*100)+r"$\alpha: $"+"{:.3f}".format(mtx_kish[i_r1, i_r2]), loc='best')
                    axs_zk4[i_r2].set(xlabel=r"$\tilde{w}$", ylabel=r"$\frac{dN}{d(\tilde{w})}$")
                    axs_zk4[i_r2].set_xticks(np.linspace(1, 10, 10))
                    # "N_0ow: {}".format(len(z3_0ow)), "N_1ow: {}".format(len(z3_1ow)), "N_2ow: {}".format(len(z3_2ow)), "N_12ow: {}".format(len(z3_12ow))
                    # "r_s: {}% \nr_x: {}% \nf_kish: {:.3f}".format(arr_r1[i_r1]*100, arr_r2[i_r2]*100, mtx_kish[i_r1, i_r2])
        # plot zw/s
        axs_ws[1].hist(x=x2*np.abs(z2), bins=bins_ws, label=r"$r_s: $"+"{}%".format(arr_r1[i_r1]*100), color=colors[i_r1], histtype='step', lw=2, alpha=0.7)
        axs_ws[1].axvline(x=mtx_xmax[i_r1, 1], color=colors[i_r1], ls='dotted', lw=2, alpha=0.7)




    ##################################################
    # plot of results
    ##################################################

    if (input=="ptitf" or input=="ptptf"):
        m_ee_train = np.sqrt( ( np.sqrt(X_train[:, 2]**2 * (1 + 1/(np.tan(X_train[:, 3]))**2)) + np.sqrt(X_train[:, 5]**2*(1 + 1/(np.tan(X_train[:, 6]))**2)) )**2 -
        ( (X_train[:, 2]/np.tan(X_train[:, 3]) + X_train[:, 5]/np.tan(X_train[:, 6]))**2 + (X_train[:, 2]*np.cos(X_train[:, 4]) + X_train[:, 5]*np.cos(X_train[:, 7]))**2 +
        (X_train[:, 2]*np.sin(X_train[:, 4]) + X_train[:, 5]*np.sin(X_train[:, 7]))**2 ) )      # invariant mass of e-e+
        
        m_ee_test = np.sqrt( ( np.sqrt(X_test[:, 2]**2 * (1 + 1/(np.tan(X_test[:, 3]))**2)) + np.sqrt(X_test[:, 5]**2*(1 + 1/(np.tan(X_test[:, 6]))**2)) )**2 -
        ( (X_test[:, 2]/np.tan(X_test[:, 3]) + X_test[:, 5]/np.tan(X_test[:, 6]))**2 + (X_test[:, 2]*np.cos(X_test[:, 4]) + X_test[:, 5]*np.cos(X_test[:, 7]))**2 +
        (X_test[:, 2]*np.sin(X_test[:, 4]) + X_test[:, 5]*np.sin(X_test[:, 7]))**2 ) )      # invariant mass of e-e+
        
        if(input=="ptitf"):
            m_ee_train = m_ee_train * X_train[:, 0] * E_cm_pro      # normalize m_e-e+ respect s_pro
            m_ee_test = m_ee_test * X_test[:, 0] * E_cm_pro
        if(input=="ptptf"):
            m_ee_train = m_ee_train * E_cm_pro      # normalize m_e-e+ respect s_pro
            m_ee_test = m_ee_test * E_cm_pro

    if (input=="p3pap"):
        m_ee_train = np.sqrt( (np.sqrt(X_train[:, 2]**2 + X_train[:, 3]**2 + X_train[:, 4]**2) + np.sqrt(X_train[:, 5]**2 + X_train[:, 6]**2 + X_train[:, 7]**2))**2 - 
        ( (X_train[:, 2] + X_train[:, 5])**2 + (X_train[:, 3] + X_train[:, 6])**2 + (X_train[:, 4] + X_train[:, 7])**2 ) )

        m_ee_test = np.sqrt( (np.sqrt(X_test[:, 2]**2 + X_test[:, 3]**2 + X_test[:, 4]**2) + np.sqrt(X_test[:, 5]**2 + X_test[:, 6]**2 + X_test[:, 7]**2))**2 - 
        ( (X_test[:, 2] + X_test[:, 5])**2 + (X_test[:, 3] + X_test[:, 6])**2 + (X_test[:, 4] + X_test[:, 7])**2 ) )
        
        m_ee_train = m_ee_train * E_cm_pro / 2     # normalize m_e-e+ respect s_pro
        m_ee_test = m_ee_test * E_cm_pro / 2

    #with PdfPages("/home/lb_linux/nn_unwgt/plot_.pdf") as pdf: 
    with PdfPages("{}/plot_{}_{}_{}_{}_{}_{}_{}_{}_{}_seed{}_{}_{}.pdf".format(path_scratch, channel, layers, input, output, lossfunc, maxfunc, output_activation, unwgt, test_sample, seed_all, dataset, where)) as pdf: 
        # plot train and input distribution
        
        if (input=="ptitf" or input=="ptptf"):
            for i_pag in range(6):

                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: {} | Learn. rate: {} \nEv_train: {} | Ev_test: {} 
Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, batch_size, learn_rate, len(X_train), len(X_test), input, output, lossfunc, seed_all), loc=9)
            
                if (i_pag==0):
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[0]))
                    axs[0].hist(x=X_train[:, 0], bins=bins_tr1, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[0]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, 0], bins=bins_tr1, weights=wgt_test, label="{} test".format(input_name[0]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, 0], bins=bins_tr1, weights=s1, label="{} pred".format(input_name[0]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[1]))
                    axs[1].hist(x=X_train[:, 1], bins=bins_beta, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[1]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, 1], bins=bins_beta, weights=wgt_test, label="{} test".format(input_name[1]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, 1], bins=bins_beta, weights=s1, label="{} pred".format(input_name[1]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel=r"$m_{e^-e^+}$", ylabel=r"$d(\sigma)/d(m_{e^-e^+})$")
                    axs[2].hist(x=m_ee_train, bins=bins_mee, weights=wgt_train/ratio_train_test, label=r"$m_{e^-e^+} train$", color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_test, bins=bins_mee, weights=wgt_test, label=r"$m_{e^-e^+} test$", color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_test, bins=bins_mee, weights=s1, label=r"$m_{e^-e^+} pred$", color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[2]))
                    axs[0].hist(x=X_train[:, i_pag*3-1], bins=bins_tr1, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[2]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, i_pag*3-1], bins=bins_tr1, weights=wgt_test, label="{} test".format(input_name[2]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, i_pag*3-1], bins=bins_tr1, weights=s1, label="{} pred".format(input_name[2]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[3]))
                    axs[1].hist(x=X_train[:, i_pag*3], bins=bins_theta, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[3]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, i_pag*3], bins=bins_theta, weights=wgt_test, label="{} test".format(input_name[3]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, i_pag*3], bins=bins_theta, weights=s1, label="{} pred".format(input_name[3]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    if (i_pag<5):
                        axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[4]))
                        axs[2].hist(x=X_train[:, i_pag*3+1], bins=bins_phi, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[4]), color='purple', histtype='step', lw=2, alpha=0.7)
                        axs[2].hist(x=X_test[:, i_pag*3+1], bins=bins_phi, weights=wgt_test, label="{} test".format(input_name[4]), color='teal', histtype='step', lw=2, alpha=0.7)
                        axs[2].hist(x=X_test[:, i_pag*3+1], bins=bins_phi, weights=s1, label="{} pred".format(input_name[4]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                        axs[2].set_yscale('log')
                        axs[2].legend(loc='best')
                    else:
                        axs[2].remove()
                
                pdf.savefig(fig)
                plt.close(fig)

        if (input=="p3pap"):
            for i_pag in range(7):
                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: {} | Learn. rate: {} \nEv_train: {} | Ev_test: {} 
Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, batch_size, learn_rate, len(X_train), len(X_test), input, output, lossfunc, seed_all), loc=9)
                if (i_pag==0):
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[0]))
                    axs[0].hist(x=X_train[:, 0], bins=bins_beta, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[0]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, 0], bins=bins_beta, weights=wgt_test, label="{} test".format(input_name[0]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, 0], bins=bins_beta, weights=s1, label="{} pred".format(input_name[0]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[1]))
                    axs[1].hist(x=X_train[:, 1], bins=bins_beta, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[1]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, 1], bins=bins_beta, weights=wgt_test, label="{} test".format(input_name[1]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, 1], bins=bins_beta, weights=s1, label="{} pred".format(input_name[1]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel=r"$m_{e^-e^+}$", ylabel=r"$d(\sigma)/d(m_{e^-e^+})$")
                    axs[2].hist(x=m_ee_train, bins=bins_mee, weights=wgt_train/ratio_train_test, label=r"$m_{e^-e^+} train$", color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_test, bins=bins_mee, weights=wgt_test, label=r"$m_{e^-e^+} test$", color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=m_ee_test, bins=bins_mee, weights=s1, label=r"$m_{e^-e^+} pred$", color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[2]))
                    axs[0].hist(x=X_train[:, i_pag*3-1], bins=bins_tr2, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[2]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, i_pag*3-1], bins=bins_tr2, weights=wgt_test, label="{} test".format(input_name[2]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[0].hist(x=X_test[:, i_pag*3-1], bins=bins_tr2, weights=s1, label="{} pred".format(input_name[2]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[3]))
                    axs[1].hist(x=X_train[:, i_pag*3], bins=bins_tr2, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[3]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, i_pag*3], bins=bins_tr2, weights=wgt_test, label="{} test".format(input_name[3]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[1].hist(x=X_test[:, i_pag*3], bins=bins_tr2, weights=s1, label="{} pred".format(input_name[3]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[4]))
                    axs[2].hist(x=X_train[:, i_pag*3+1], bins=bins_tr2, weights=wgt_train/ratio_train_test, label="{} train".format(input_name[4]), color='purple', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=X_test[:, i_pag*3+1], bins=bins_tr2, weights=wgt_test, label="{} test".format(input_name[4]), color='teal', histtype='step', lw=2, alpha=0.7)
                    axs[2].hist(x=X_test[:, i_pag*3+1], bins=bins_tr2, weights=s1, label="{} pred".format(input_name[4]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                pdf.savefig(fig)
                plt.close(fig)


        # plot wgt mean respect inputs
        if (input=="ptitf" or input=="ptptf"):
            for i_pag in range(6):

                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: {} | Learn. rate: {} \nEv_train: {} | Ev_test: {} 
    Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, batch_size, learn_rate, len(X_train), len(X_test), input, output, lossfunc, seed_all), loc=9)

                if (i_pag==0):
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel="mean wgt({})".format(input_name[0]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 0]>=bins_tr1[i] and X_train[j, 0]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, 0]>=bins_tr1[i] and X_test[j, 0]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, 0]>=bins_tr1[i] and X_test[j, 0]<bins_tr1[i+1]]) for i in range(49)]
                    axs[0].plot(bins_tr1[1:], w_mean_tr, label="{} train".format(input_name[0]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_va, label="{} test".format(input_name[0]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_pr, label="{} pred".format(input_name[0]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[1]))
                    w_tr, w_va, w_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 1]>=bins_beta[i] and X_train[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, 1]>=bins_beta[i] and X_test[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, 1]>=bins_beta[i] and X_test[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_beta[1:], w_mean_tr, label="{} train".format(input_name[1]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_va, label="{} test".format(input_name[1]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_pr, label="{} pred".format(input_name[1]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel=r"$m_{e^-e^+}$", ylabel=r"$d(\sigma)/d(m_{e^-e^+})$")
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if m_ee_train[j]>=bins_mee[i] and m_ee_train[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if m_ee_test[j]>=bins_mee[i] and m_ee_test[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if m_ee_test[j]>=bins_mee[i] and m_ee_test[j]<bins_mee[i+1]]) for i in range(49)]
                    axs[2].plot(bins_mee[1:], w_mean_tr, label="{} train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_va, label="{} test".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_pr, label="{} pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[2]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3-1]>=bins_tr1[i] and X_train[j, i_pag*3-1]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, i_pag*3-1]>=bins_tr1[i] and X_test[j, i_pag*3-1]<bins_tr1[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, i_pag*3-1]>=bins_tr1[i] and X_test[j, i_pag*3-1]<bins_tr1[i+1]]) for i in range(49)]
                    axs[0].plot(bins_tr1[1:], w_mean_tr, label="{} train".format(input_name[2]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_va, label="{} test".format(input_name[2]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_tr1[1:], w_mean_pr, label="{} pred".format(input_name[2]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[3]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3]>=bins_theta[i] and X_train[j, i_pag*3]<bins_theta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, i_pag*3]>=bins_theta[i] and X_test[j, i_pag*3]<bins_theta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, i_pag*3]>=bins_theta[i] and X_test[j, i_pag*3-1]<bins_theta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_theta[1:], w_mean_tr, label="{} train".format(input_name[3]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_theta[1:], w_mean_va, label="{} test".format(input_name[3]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_theta[1:], w_mean_pr, label="{} pred".format(input_name[3]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    if (i_pag<5):
                        axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[4]))
                        w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                        w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3+1]>=bins_phi[i] and X_train[j, i_pag*3+1]<bins_phi[i+1]]) for i in range(49)]
                        w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, i_pag*3+1]>=bins_phi[i] and X_test[j, i_pag*3+1]<bins_phi[i+1]]) for i in range(49)]
                        w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, i_pag*3+1]>=bins_phi[i] and X_test[j, i_pag*3+1]<bins_phi[i+1]]) for i in range(49)]
                        axs[2].plot(bins_phi[1:], w_mean_tr, label="{} train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                        axs[2].plot(bins_phi[1:], w_mean_va, label="{} test".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                        axs[2].plot(bins_phi[1:], w_mean_pr, label="{} pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                        axs[2].set_yscale('log')
                        axs[2].legend(loc='best')
                    else:
                        axs[2].remove()

                pdf.savefig(fig)
                plt.close(fig)

        if (input=="p3pap"):
            for i_pag in range(7):

                fig, axs = plt.subplots(3, figsize=(8.27, 11.69)) 

                fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: {} | Learn. rate: {} \nEv_train: {} | Ev_test: {} 
    Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {}""".format(len(X_train[0]), layers,
                n_epochs, batch_size, learn_rate, len(X_train), len(X_test), input, output, lossfunc, seed_all), loc=9)

                if (i_pag==0):
                    axs[0].set(xlabel="{}".format(input_name[0]), ylabel="mean wgt({})".format(input_name[0]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 0]>=bins_beta[i] and X_train[j, 0]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, 0]>=bins_beta[i] and X_test[j, 0]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, 0]>=bins_beta[i] and X_test[j, 0]<bins_beta[i+1]]) for i in range(49)]
                    axs[0].plot(bins_beta[1:], w_mean_tr, label="{} train".format(input_name[0]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_va, label="{} test".format(input_name[0]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_pr, label="{} pred".format(input_name[0]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}".format(input_name[1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[1]))
                    w_tr, w_va, w_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, 1]>=bins_beta[i] and X_train[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, 1]>=bins_beta[i] and X_test[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, 1]>=bins_beta[i] and X_test[j, 1]<bins_beta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_beta[1:], w_mean_tr, label="{} train".format(input_name[1]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_va, label="{} test".format(input_name[1]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_pr, label="{} pred".format(input_name[1]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel=r"$m_{e^-e^+}$", ylabel=r"$d(\sigma)/d(m_{e^-e^+})$")
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if m_ee_train[j]>=bins_mee[i] and m_ee_train[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if m_ee_test[j]>=bins_mee[i] and m_ee_test[j]<bins_mee[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if m_ee_test[j]>=bins_mee[i] and m_ee_test[j]<bins_mee[i+1]]) for i in range(49)]
                    axs[2].plot(bins_mee[1:], w_mean_tr, label="{} train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_va, label="{} test".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                    axs[2].plot(bins_mee[1:], w_mean_pr, label="{} pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                    axs[2].axvline(x=91.2, color='black', ls='--', lw=0.5)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                if (i_pag>0):
                    axs[0].set(xlabel="{}({})".format(input_name[2], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[2]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3-1]>=bins_beta[i] and X_train[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, i_pag*3-1]>=bins_beta[i] and X_test[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, i_pag*3-1]>=bins_beta[i] and X_test[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    axs[0].plot(bins_beta[1:], w_mean_tr, label="{} train".format(input_name[2]), color='purple', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_va, label="{} test".format(input_name[2]), color='teal', marker='.', alpha=0.7)
                    axs[0].plot(bins_beta[1:], w_mean_pr, label="{} pred".format(input_name[2]), color='goldenrod', marker='.', alpha=0.7)
                    axs[0].set_yscale('log')
                    axs[0].legend(loc='best')

                    axs[1].set(xlabel="{}({})".format(input_name[3], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[3]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3]>=bins_beta[i] and X_train[j, i_pag*3]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, i_pag*3]>=bins_beta[i] and X_test[j, i_pag*3]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, i_pag*3]>=bins_beta[i] and X_test[j, i_pag*3-1]<bins_beta[i+1]]) for i in range(49)]
                    axs[1].plot(bins_beta[1:], w_mean_tr, label="{} train".format(input_name[3]), color='purple', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_va, label="{} test".format(input_name[3]), color='teal', marker='.', alpha=0.7)
                    axs[1].plot(bins_beta[1:], w_mean_pr, label="{} pred".format(input_name[3]), color='goldenrod', marker='.', alpha=0.7)
                    axs[1].set_yscale('log')
                    axs[1].legend(loc='best')

                    axs[2].set(xlabel="{}({})".format(input_name[4], part_name[i_pag-1]), ylabel=r"$d(\sigma)$"+"/d({}) [pb]".format(input_name[4]))
                    w_mean_tr, w_mean_va, w_mean_pr = np.zeros(49), np.zeros(49), np.zeros(49)
                    w_mean_tr = [np.mean([wgt_train[j] for j in range(len(wgt_train)) if X_train[j, i_pag*3+1]>=bins_beta[i] and X_train[j, i_pag*3+1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_va = [np.mean([wgt_test[j] for j in range(len(wgt_test)) if X_test[j, i_pag*3+1]>=bins_beta[i] and X_test[j, i_pag*3+1]<bins_beta[i+1]]) for i in range(49)]
                    w_mean_pr = [np.mean([s1[j] for j in range(len(s1)) if X_test[j, i_pag*3+1]>=bins_beta[i] and X_test[j, i_pag*3+1]<bins_beta[i+1]]) for i in range(49)]
                    axs[2].plot(bins_beta[1:], w_mean_tr, label="{} train".format(input_name[4]), color='purple', marker='.', alpha=0.7)
                    axs[2].plot(bins_beta[1:], w_mean_va, label="{} test".format(input_name[4]), color='teal', marker='.', alpha=0.7)
                    axs[2].plot(bins_beta[1:], w_mean_pr, label="{} pred".format(input_name[4]), color='goldenrod', marker='.', alpha=0.7)
                    axs[2].set_yscale('log')
                    axs[2].legend(loc='best')

                pdf.savefig(fig)
                plt.close(fig)

        fig, axs = plt.subplots(2, figsize=(8.27, 11.69)) 

        # plot ws
        
        #fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: {} | Learn. rate: {} \nEv_train: {} | Ev_test: {} 
            #Normalization: {} | Output: {} \nLoss: {} | Seed tf and np: {} \nTime init : {:.3f}s 
            #Time pred : {:.3f}s | Time unwgt1 : {:.3f}s | Time unwgt2 : {:.3f}s""".format(len(X_train[0]), layers,
            #n_epochs, batch_size, learn_rate, len(X_train), len(X_test), input, output, lossfunc, seed_all, time_init, time_pred, time_unwgt1, time_unwgt2))
        
        #fig.legend(title = """Layers: {}, {}, 1 \nEpochs: {} | Batch size: {} | Learn. rate: {} \nEv_train: {} | Ev_test: {} \nNormalization: {} | Output: {} \nLoss: {} | Seed tf and np: {} \nTime pred: {}""".format(len(X_train[0]), layers, n_epochs, batch_size, learn_rate, len(X_train), len(X_test), input, output, lossfunc, seed_all, time_pred))
        fig.legend(title=plot_legend)

        x1 = np.divide(wgt_test, s1)
        if (output=="wgt"):
            axs_ws[0].set_xscale('log')

        # plot predicted output
        axs[0].set(xlabel="output normalized", ylabel="dN/dw")
        axs[0].hist(x=wgt_train_pred, bins=bins_w_pred, label="w train", weights=(1/ratio_train_test)*np.ones_like(wgt_train_pred), color='teal', histtype='step', lw=3, alpha=0.5)
        axs[0].hist(x=wgt_test_pred, bins=bins_w_pred, label="w test", color='darkblue', histtype='step', lw=3, alpha=0.5)
        axs[0].hist(x=s1_pred, bins=bins_w_pred, label="s pred", color='purple', histtype='step', lw=3, alpha=0.5)
        if (minpred!="nmp"):
            axs[0].axvline(x=s_low_pred, label="s_low_pred: {:.2e}".format(s_low_pred), color='darkred', linewidth=2, linestyle='dotted')
        axs[0].legend(loc=3)
        if (output=="wno" or output=="lno" or output=="wgt" or output=="rww"):
            axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        
        axs[1].set(xlabel="w [pb]", ylabel="w/s")
        h2 = axs[1].hist2d(wgt_test, x1, bins=[bins_w, bins_ws], norm=mpl.colors.LogNorm())
        axs[1].axhline(y=1, color='orange', linewidth=1, linestyle='dotted')
        if (minpred!="nmp"):
            axs[1].axvline(x=s_low, label="s_low: {:.2e}".format(s_low), color='darkred', linewidth=2, linestyle='dotted')
            axs[1].legend(loc='best')
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        plt.colorbar(h2[3], ax=axs[1], label="Frequency") 
        axs[1].legend(loc='best')

        pdf.savefig(fig)
        plt.close(fig)

        axs_ws[0].set(xlabel="w/s", ylabel="dN/d(w/s)")
        axs_ws[0].hist(x1, bins=bins_ws)
        axs_ws[0].axvline(x=1, color='black', ls='--')
        axs_ws[0].set_xscale('log')
        axs_ws[0].set_xticks([10**(-6), 10**(-4), 10**(-2), 10**(0), 10**(2), 10**(4), 10**(6)])
        axs_ws[0].set_yscale('log')

        axs_ws[1].legend(loc='best')
        axs_ws[1].set(xlabel=r"$\tilde{w}^{(1)} \cdot w/s$", ylabel=r"$dN/d(\tilde{w}^{(1)} \cdot w/s)$")
        axs_ws[1].axvline(x=1, color='black', ls='--')
        axs_ws[1].set_xscale('log')
        axs_ws[1].set_xticks([10**(-6), 10**(-4), 10**(-2), 10**(0), 10**(2), 10**(4), 10**(6)])
        axs_ws[1].set_yscale('log')

        pdf.savefig(fig_ws)
        plt.close(fig_ws)


        # plot eff
        fig, axs = plt.subplots(2, figsize=(8.27, 11.69)) 

        if (unwgt=="new"):
            axs[0].set(xlabel=r"$s_{thres}$ [pb]", ylabel=r"$\varepsilon_1$")
            axs[0].plot(arr_smax, arr_eff1, marker='.', markersize=15)
        if (unwgt=="pap"):
            axs[0].set(xlabel=r"$w_{thres}$ [pb]", ylabel=r"$\varepsilon_1$")
            axs[0].plot(arr_wmax, arr_eff1, marker='.', markersize=15)
        axs[0].set_title("efficiency 1", fontsize=16)
        
        axs[1].set(xlabel=r"$x_{thres}$", ylabel=r"$\varepsilon_2$")
        if (unwgt=="new"):
            for i in range(len(arr_r2)):                      # plot the curve of the efficiencies in function of the x_max with fixed s_max
                axs[1].plot(mtx_xmax[i, :], mtx_eff2[i], marker='.', markersize=15, color=colors[i], label=r"$r_s: $"+"{}%".format(arr_r1[i]*100))
        if (unwgt=="pap"):
            for i in range(len(arr_r2)):                      # plot the curve of the efficiencies in function of the x_max with fixed s_max
                axs[1].plot(mtx_xmax[i, :], mtx_eff2[i], marker='.', markersize=15, color=colors[i], label=r"$r_s: $"+"{}%".format(arr_r1[i]*100))
        axs[1].legend(loc='best')
        axs[1].set_title("efficiency 2", fontsize=16)
        pdf.savefig(fig)
        plt.close(fig)


        fig, axs = plt.subplots(2, figsize=(8.27, 11.69)) 

        # plot feff
        for i in range(len(arr_r1)):                      # x, y labels of the f_eff colormap
            axs[0].text(-0.70, 0.45+i, r"$r_s: $"+"{}%".format(arr_r1[i]*100), fontsize = 12)
            axs[0].text(0.25+i, -0.25, r"$r_x: $"+"{}%".format(arr_r2[i]*100), fontsize = 12)
        #axs[0].axis('off')
        axs[0].set_title(label="effective gain factor", fontsize=16)
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        h1 = axs[0].pcolormesh(mtx_feff, cmap=cmap)
        ticks_feff = np.linspace(np.min(mtx_feff), np.max(mtx_feff), 10, endpoint=True)
        plt.colorbar(h1, ax=axs[0], ticks=ticks_feff, label=r"$f_{eff}$")

        # plot of plR (plot 1/R) 
        x_plR = np.linspace(1, len(arr_r1)*len(arr_r2), len(arr_r1)*len(arr_r2))
        y1_plR = [ 1 / ( arr_eff1[i//len(arr_r1)] * ( mtx_kish[i//len(arr_r1),i%len(arr_r2)] * mtx_eff2[i//len(arr_r1),i%len(arr_r2)] / (eff1_st*eff2_st) - 1 ) ) for i in range(len(arr_r1)*len(arr_r2))] 
        y5_plR = [ 1 / ( arr_eff1[i//len(arr_r1)] * ( mtx_kish[i//len(arr_r1),i%len(arr_r2)] * mtx_eff2[i//len(arr_r1),i%len(arr_r2)] / (eff1_st*eff2_st*5) - 1 ) ) for i in range(len(arr_r1)*len(arr_r2))] 
        #y10_plR = [ 1 / ( arr_eff1[i//len(arr_r1)] * ( mtx_kish[i//len(arr_r1),i%len(arr_r2)] * mtx_eff2[i//len(arr_r1),i%len(arr_r2)] / (eff1_st*eff2_st*10) - 1 ) ) for i in range(len(arr_r1)*len(arr_r2))] 
        # bar color with the kish factor
        kish_dif = np.max(mtx_kish) - np.min(mtx_kish) 
        c_plR = [cmap((mtx_kish[i//len(arr_r1), i%len(arr_r2)]-np.min(mtx_kish))/kish_dif) for i in range(len(arr_r1)*len(arr_r2))]
        norm_plR = mpl.colors.Normalize(vmin=np.min(mtx_kish), vmax=np.max(mtx_kish)) 
        sm_plR = plt.cm.ScalarMappable(cmap=cmap, norm=norm_plR)
        sm_plR.set_array([])
        axs[1].scatter(x_plR, y1_plR, color=c_plR, s=100, alpha=0.7, marker='o', label=r"$f_{eff}>1$")
        axs[1].scatter(x_plR, y5_plR, color=c_plR, s=100, alpha=0.7, marker='X', label=r"$f_{eff}>5$")
        #axs[1].scatter(x_plR, y10_plR, color=c_plR, s=100, alpha=0.7, marker='X', label="f_eff>10")
        x_plR_min, x_plR_max = axs[1].get_xlim()
        #axs[1].axhline(y=t_ratio, label="R \n(min value for positive gain)", color='green')
        plt.colorbar(sm_plR, ax=axs[1], ticks=np.linspace(np.min(mtx_kish), np.max(mtx_kish), 5), label=r"$\alpha$")
        axs[1].text(-0.5, -0.7 , r"$r_s: $"+"\n"+r"$r_x: $", fontsize = 6)
        axs[1].set_xticks(x_plR)
        axs[1].set_xticklabels(["{}% \n{}%".format(arr_r1[i//len(arr_r1)]*100,arr_r2[i%len(arr_r2)]*100) for i in range(len(arr_r1)*len(arr_r2))])
        axs[1].tick_params(axis='x', which='major', labelsize=6)
        axs[1].set_ylim([10**(-3), 10**4])
        #axs[1].set_ylabel("Overwgt unwgt 1, 2")
        axs[1].set_yscale('log')
        axs[1].set_ylabel(r"$\left(\frac{\langle t_{st} \rangle}{\langle t_{nn} \rangle}\right)_{min}$")
        axs[1].legend(loc='best')
        pdf.savefig(fig)
        plt.close(fig)

        
        # plot table
        table = axs_tab.table(cellText = data_table, cellLoc ='center', loc ='upper left' )     
        for k in range(row_max):
            for r in range(0, len(col_lab)):
                cell = table[k, r]
                cell.set_height(0.06)
        table.set_fontsize(7)
        pdf.savefig(fig_tab)
        plt.close(fig_tab)


        # unweighting of standard sample with fixed kish factor

        def effective_gain_st(eff1, eff2, eff_st, t_su):       # effective gain factor where the standard method computes all the matrix elements
            #t_ratio = 0.002
            res = 1 / ((t_su/t_st)*eff_st/(eff1*eff2) + eff_st/eff2)
            return res
        
        mtx_eff_st, mtx_feff_st =  np.zeros((len(arr_r1), len(arr_r2))), np.zeros((len(arr_r1), len(arr_r2)))
        rand_st = np.random.rand(len(s1))
        computed_kish_st = np.full((len(arr_r1), len(arr_r2)), False)
        for i_r1 in range(len(arr_r1)):               # double loop over all r1, r2 combinations 
            for i_r2 in range(len(arr_r2)):
                if (mtx_kish[i_r1, i_r2]<1):             # if kish_su<1, we require r2_st such that kish_st=kish_su
                    kish_st = 0 
                    r = (1 - mtx_kish[i_r1, i_r2]**2) / 4      # rough ad-hoc formula
                    for j in range(50): 
                        n_kept = 0                             # number of kept events
                        w_st_max = my_max(wgt_test, r_ow=r)
                        w2_st = np.empty(len(wgt_test))
                        z2_st = np.empty(len(wgt_test))
                        for i in range(len(wgt_test)):                 # second unweighting
                            if (wgt_test[i]>(rand_st[i]*w_st_max)):
                                w2_st[n_kept] = wgt_test[i]
                                wgt_z = np.maximum(1, wgt_test[i]/w_st_max)
                                z2_st[n_kept] = wgt_z
                                n_kept += 1
                        w2_st = w2_st[0:n_kept]
                        z2_st = z2_st[0:n_kept]
                        kish_st = f_kish(z2_st)
                        if (np.abs(mtx_kish[i_r1, i_r2]-kish_st)/mtx_kish[i_r1, i_r2] <= 0.01):
                            computed_kish_st[i_r1, i_r2] = True
                            break
                        else:                            # if kish_st!=kish_su we modify r and perform again the second unweighting
                            r = np.abs( r + (kish_st - mtx_kish[i_r1, i_r2])/8) 
                else:                                    # if kish_su=1 we use the r2_st=r2_su
                    n_kept = 0                             # number of kept events
                    w_st_max = my_max(wgt_test, r_ow=0)
                    w2_st = np.empty(len(wgt_test))
                    z2_st = np.empty(len(wgt_test))
                    for i in range(len(wgt_test)):                 # second unweighting
                        if (wgt_test[i]>(rand_st[i]*w_st_max)):
                            w2_st[n_kept] = wgt_test[i]
                            wgt_z = np.maximum(1, wgt_test[i]/w_st_max)
                            z2_st[n_kept] = wgt_z
                            n_kept += 1
                    w2_st = w2_st[0:n_kept]
                    z2_st = z2_st[0:n_kept]
                    computed_kish_st[i_r1, i_r2] = True
                if(computed_kish_st[i_r1, i_r2]==True):
                    mtx_eff_st[i_r1, i_r2] = efficiency(wgt_test, w_st_max)
                    mtx_feff_st[i_r1, i_r2] = effective_gain_st(arr_eff1[i_r1], mtx_eff2[i_r1, i_r2], mtx_eff_st[i_r1, i_r2], t_nn)
                else:
                    mtx_eff_st[i_r1, i_r2] = 0
                    mtx_feff_st[i_r1, i_r2] = 0

        fig_st, axs_st = plt.subplots(2, figsize=(8.27, 11.69))
        for i in range(len(arr_r1)):                      # y labels of the f_eff colormap
            if (unwgt=="new"):
                axs_st[0].text(-0.5, 0.3+i, "$r_s: $"+"{}%".format(arr_r1[i]*100), fontsize = 8)
            if (unwgt=="pap"):
                axs_st[0].text(-0.5, 0.3+i, "$r_s: $"+"{}%".format(arr_r1[i]*100), fontsize = 8)
        for i in range(len(arr_r2)):                      # x labels of the f_eff colormap
            axs_st[0].text(0.3+i, -0.3, "$r_x: $"+"{}%".format(arr_r2[i]*100), fontsize = 8)
        #axs[0].axis('off')
        axs_st[0].set_title(label="effective gain factor with fixed Kish factor")
        axs_st[0].set_yticklabels([])
        axs_st[0].set_xticklabels([])
        h1 = axs_st[0].pcolormesh(mtx_feff_st, cmap=cmap)
        ticks_feff = np.linspace(np.min(mtx_feff_st), np.max(mtx_feff_st), 10, endpoint=True)
        plt.colorbar(h1, ax=axs_st[0], ticks=ticks_feff)

        axs_st[1].set(xlabel="r_x", ylabel="MG efficiency with fixed kished factor")
        for i in range(len(arr_r2)):                      # plot the curve of the efficiencies in function of the x_max with fixed s_max
            axs_st[1].plot(np.linspace(1, len(arr_r2), len(arr_r2)), mtx_eff_st[i], marker='.', markersize=15, color=colors[i], label="$s_{thres} $"+"{}%".format(arr_r1[i]*100))
            axs_st[1].text(0.3+i, -0.3, "$r_x: $"+"{}%".format(arr_r2[i]*100), fontsize = 8)
        axs_st[1].set_xticklabels([])
        axs_st[1].legend(loc='best')

        pdf.savefig(fig_st)
        plt.close(fig_st)


        # plot zk
        pdf.savefig(fig_zk1)
        plt.close(fig_zk1)
        pdf.savefig(fig_zk2)
        plt.close(fig_zk2)
        pdf.savefig(fig_zk3)
        plt.close(fig_zk3)
        pdf.savefig(fig_zk4)
        plt.close(fig_zk4)


    print("\n--------------------------------------------------")
    print(plot_legend)


    
    """
    # plot data types accuracy
    fig, axs = plt.subplots(3, figsize=(8.27, 11.69))
    axs[0].set_yscale('log')
    axs[0].set_ylim([10**(-31), 10**(-1)])
    axs[0].set(xlabel="{}".format(input_name[0]), ylabel="dN/d({})".format(input_name[0]))
    axs[0].hist(x=X_train[:, 0], bins=bins_tr1, weights=wgt_train/ratio_train_test, label="{}_train".format(input_name[0]), color='purple', histtype='step', lw=2, alpha=0.7)
    axs[0].hist(x=X_val[:, 0], bins=bins_tr1, weights=wgt_val, label="{}_val".format(input_name[0]), color='teal', histtype='step', lw=2, alpha=0.7)
    axs[0].hist(x=X_val[:, 0], bins=bins_tr1, weights=s1, label="{}_pred".format(input_name[0]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
    axs[0].legend(loc='best')

    s1_1 = np.double(s1)
    wgt_val_1 = np.double(wgt_val)
    wgt_train_1 = np.double(wgt_train)
    axs[1].set_yscale('log')
    axs[1].set_ylim([10**(-31), 10**(-1)])
    axs[1].set(xlabel="{}".format(input_name[0]), ylabel="dN/d({})".format(input_name[0]))
    axs[1].hist(x=X_train[:, 0], bins=bins_tr1, weights=wgt_train_1/ratio_train_test, label="{}_train np double".format(input_name[0]), color='purple', histtype='step', lw=2, alpha=0.7)
    axs[1].hist(x=X_val[:, 0], bins=bins_tr1, weights=wgt_val_1, label="{}_val np double".format(input_name[0]), color='teal', histtype='step', lw=2, alpha=0.7)
    axs[1].hist(x=X_val[:, 0], bins=bins_tr1, weights=s1_1, label="{}_pred np double".format(input_name[0]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
    axs[1].legend(loc='best')

    s1_2 = np.longdouble(s1)
    wgt_val_2 = np.longdouble(wgt_val)
    wgt_train_2 = np.longdouble(wgt_train)
    axs[2].set_yscale('log')
    axs[2].set_ylim([10**(-31), 10**(-1)])
    axs[2].set(xlabel="{}".format(input_name[0]), ylabel="dN/d({})".format(input_name[0]))
    axs[2].hist(x=X_train[:, 0], bins=bins_tr1, weights=wgt_train_2/ratio_train_test, label="{}_train np longdouble".format(input_name[0]), color='purple', histtype='step', lw=2, alpha=0.7)
    axs[2].hist(x=X_val[:, 0], bins=bins_tr1, weights=wgt_val_2, label="{}_val np longdouble".format(input_name[0]), color='teal', histtype='step', lw=2, alpha=0.7)
    axs[2].hist(x=X_val[:, 0], bins=bins_tr1, weights=s1_2, label="{}_pred np longdouble".format(input_name[0]), color='goldenrod', histtype='step', lw=2, alpha=0.7)
    axs[2].legend(loc='best')

    fig.savefig("/home/lb_linux/nn_unwgt/G128/plot_EoverE.pdf", format='pdf')
    plt.close(fig)
    """



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
