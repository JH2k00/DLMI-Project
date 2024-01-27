import numpy as np
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg

def calc_allfeatures(X):
    features = []
    for i in range(X.shape[0]):
        data = X[i, :, :]
        f, Pxx = calc_Periodogram(data)
        features.extend([calc_MAV(data),calc_MAV1(data),calc_SSI(data),calc_VAR(data),calc_TM(data,2),calc_RMS(data),calc_V(data,4),calc_LOG(data),calc_WL(data),calc_AAC(data),calc_DASDV(data),calc_MYOP(data,0.1),calc_WAMP(data,0.1),calc_SSC(data,0.002),calc_MNF(f,Pxx),calc_MDF(f,Pxx),calc_PKF(f,Pxx),calc_MNP(f,Pxx),calc_SM(f,Pxx,3),calc_FR(f, Pxx, [30,150,150,500]),calc_PSR(f,Pxx,5),calc_VCF(f,Pxx)])
        features.extend(list(calc_AR_coeff(data, 4)))
        features.extend(list(calc_hist(data,4)))
    return np.array(features)
     
def calc_MAV(data):
    return np.mean(np.abs(data))

def calc_MAV1(data):
    N = data.size
    return (np.sum(np.abs(data[int(0.25*N):int(0.75*N), int(0.25*N):int(0.75*N)])) + np.sum(np.abs(data[:int(0.25*N), :int(0.25*N)]))*0.5 + np.sum(np.abs(data[int(0.75*N):, int(0.75*N):]))*0.5) / N

def calc_SSI(data):
    return(np.sum(np.square(data)))

def calc_VAR(data):
    return np.var(data, ddof=1)

def calc_TM(data, order):
    return np.abs(np.mean(np.power(data,order)))

def calc_RMS(data):
    return np.sqrt(np.mean(np.square(data)))

def calc_V(data, order):
    return np.power(np.mean(np.power(data,order)),1/order)

def calc_LOG(data):
    return np.exp(np.mean(np.log(np.abs(data) + np.finfo(float).eps)))

def calc_WL(data):
    return np.sum(np.abs(np.diff(data)))

def calc_AAC(data):
    return np.mean(np.abs(np.diff(data)))

def calc_DASDV(data):
    return np.sqrt(np.mean(np.square(np.diff(data))))

def calc_MYOP(data, th):
    stand_data = (data - np.mean(data)) / np.std(data) #Standardize for a single th
    return np.mean((stand_data >= th))

def calc_WAMP(data, th):
    stand_data = (data - np.mean(data)) / np.std(data) #Standardize for a single th
    diff= np.abs(np.diff(stand_data))
    return np.sum((diff >= th))

def calc_SSC(data, th):
    stand_data_diff = np.diff((data - np.mean(data)) /np.std(data)) #Standardize for a single th
    return np.sum((-stand_data_diff[:-1]*stand_data_diff[1:]) >= th) 

#Remove mean because 0 Hz is not interesting
def calc_Periodogram(data):
    data = data.flatten()
    f, Pxx_den = signal.periodogram(data-np.mean(data), fs=1000, detrend=False, scaling='spectrum') 
    return f, Pxx_den

def calc_MNF(f, Pxx):
    return np.dot(f, Pxx) / np.sum(Pxx)

def calc_MDF(f, Pxx):
    energy_cumsum = np.cumsum(Pxx)
    return f[np.where(energy_cumsum>np.max(energy_cumsum)/2)[0][0]] 

def calc_PKF(f, Pxx):
    return f[np.argmax(Pxx)]

def calc_MNP(f, Pxx):
    return np.mean(Pxx)

def calc_SM(f, Pxx, order):
    return np.dot(Pxx,np.power(f, order))

def calc_FR(f, Pxx, frequencies):
    #frequencies must be an array / a list with : [lower freq min, lower freq max, higher freq min, higher freq max]
    low_freq = np.sum(Pxx[np.logical_and(f>=frequencies[0], f<=frequencies[1])])
    high_freq = np.sum(Pxx[np.logical_and(f>=frequencies[2], f<=frequencies[3])])
    return low_freq / high_freq

def calc_PSR(f, Pxx, n):
    max_P = np.argmax(Pxx)
    local_en = np.sum(Pxx[max((0,max_P-n)) : min((len(Pxx)-1,max_P + n))])
    return local_en / np.sum(Pxx)

def calc_VCF(f, Pxx):
    SM0 = calc_SM(f, Pxx, 0)
    return (calc_SM(f, Pxx, 2) / SM0) - (calc_SM(f, Pxx, 1) / SM0)**2

#Output is an array of length ncoeff
def calc_AR_coeff(data, n_coeff): 
    model = AutoReg(data.flatten(), lags=10) # train autoregression
    model_fit = model.fit()
    return np.array(model_fit.params[:n_coeff])

#Output is an array of length bins 
def calc_hist(data, bins):
    return np.histogram(data, bins)[0]