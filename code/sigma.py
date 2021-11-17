#a python file to import the SIGMA algorithm for glottal instants extraction
#pass as arguments (egg_signal,sample_rate_of_egg) to get_glottal,
#output is locations of gci and goi within given egg signal

import numpy as np
import librosa
import pywt
from sklearn.mixture import GaussianMixture
# from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt 


def sliding_window_view(signal, window_shape):
    y = []
    l = signal.shape[0]

    for i in range(0, l - window_shape):
        y.append(signal[i:i+window_shape])

    return np.asarray(y)

def energy_weighted_group_delay(signal,sr_egg):
    
    R = int(0.0025*sr_egg)
    mult = np.arange(R)
        
    # X = np.lib.stride_tricks.sliding_window_view(signal,window_shape = (R))
    X = sliding_window_view(signal, R)
    X_r = X*mult
    
    X = np.fft.fft(X,axis = 1)
    X_r = np.fft.fft(X_r,axis = 1)    
    
    group_delay = np.real(X_r/X)
    X_sq = np.square(np.abs(X))
    ewgd = np.sum(X_sq*group_delay,axis = 1)/np.sum(X_sq,axis = 1)
    
    ewgd -= (R - 1)/2.
    
    return ewgd

def zero_crossing_pos2neg(signal):
    rectified = signal > 0
    return np.where(np.logical_and(np.logical_xor(rectified[:-1],rectified[1:]),rectified[:-1]))[0]

def get_cluster(zc_pos2neg,ewgd,p_positive,sr_egg):
    feature_mat = np.zeros((len(zc_pos2neg),3))

    R = int(0.0025*sr_egg)
    
    ideal = np.arange(R//2,-(R+1)//2,-1)
        
    for i in range(len(zc_pos2neg)):
        ewgd_window = ewgd[zc_pos2neg[i] - int((R-1)/2):zc_pos2neg[i] + int((R-1)/2) + 1]
        feature_mat[i,0] = np.sqrt(np.mean(np.square(ewgd_window - ideal[:len(ewgd_window)])))
        p_pos_window = p_positive[zc_pos2neg[i] - int((R-1)/2):zc_pos2neg[i] + int((R-1)/2) + 1]
        feature_mat[i,1] = np.amax(p_pos_window**(1/3.))
        feature_mat[i,2] = np.sum(p_pos_window**(1/3.))
        
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature_mat)
    label = gmm.predict(feature_mat)
    
    if np.mean(feature_mat[label == 1,2]) > np.mean(feature_mat[label == 0,2]):
        return zc_pos2neg[label == 1]
    return zc_pos2neg[label == 0]

def swallowing(gci,sr_egg):
    N_max = 0.02 * sr_egg
    diff = gci[1:] - gci[:-1]
    keep = np.zeros(len(gci))
    remove = diff>N_max
    keep[:-1] += remove
    keep[1:] += remove
    return gci[keep == 0]

def goi_post_processing(goi_candidates,gci,sr_egg):
    goi = []
    N_max = 0.02 * sr_egg
    for i in range(len(gci)-1):
        if gci[i+1] - gci[i] < N_max:
            check = 0
            for j in range(gci[i] + int((gci[i+1]-gci[i])*0.1),gci[i] + int((gci[i+1]-gci[i])*0.9)):
                if j in goi_candidates:
                    goi.append(j)
                    check = 1
                    break
            if check == 0:
                goi.append(gci[i] + int((gci[i+1]-gci[i])*0.5))
    return goi 

def get_glottal(egg,sr_egg):
    
    if len(egg)%8 != 0:
        egg = egg[:-(len(egg)%8)]
    
    swt = pywt.swt(egg, wavelet = "bior1.5", level = 3)
    multiscale_product = swt[0][1]*swt[1][1]*swt[2][1]
    
    p_positive = multiscale_product*(multiscale_product>0)
    p_negative = multiscale_product*(multiscale_product<0)
    
    ewgd_gci = energy_weighted_group_delay(p_positive,sr_egg)
    ewgd_gci[np.where(np.isnan(ewgd_gci))] = 0
    
    ewgd_goi = energy_weighted_group_delay(-p_negative,sr_egg)
    ewgd_goi[np.where(np.isnan(ewgd_goi))] = 0
    
    zc_pos2neg_gci = zero_crossing_pos2neg(ewgd_gci)
    zc_pos2neg_goi = zero_crossing_pos2neg(ewgd_goi)
    
    R = int(0.0025*sr_egg)

    for i in range(len(zc_pos2neg_gci)):
        if zc_pos2neg_gci[i] > int((R-1)/2):
            zc_pos2neg_gci = zc_pos2neg_gci[i:]
            break
            
    for i in range(len(zc_pos2neg_goi)):
        if zc_pos2neg_goi[i] > int((R-1)/2):
            zc_pos2neg_goi = zc_pos2neg_goi[i:]
            break
            
    cluster_gci = get_cluster(zc_pos2neg_gci,ewgd_gci,p_positive,sr_egg)
    cluster_goi = get_cluster(zc_pos2neg_goi,ewgd_goi,-p_negative,sr_egg)
    
    gci = swallowing(cluster_gci,sr_egg)
    
    goi = goi_post_processing(cluster_goi,gci,sr_egg)
    
    return gci,goi

def naylor_metrics(ref_signal, est_signal):

    assert (np.squeeze(ref_signal).ndim == 1)
    assert (np.squeeze(est_signal).ndim == 1)

    ref_signal = np.squeeze(ref_signal)
    est_signal = np.squeeze(est_signal)

    min_f0 = 50
    max_f0 = 500
    min_glottal_cycle = 1 / max_f0
    max_glottal_cycle = 1 / min_f0

    nHit = 0
    nMiss = 0
    nFalse = 0
    nCycles = 0
    highNumCycles = 100000
    estimation_distance = np.full(highNumCycles, np.nan)

    ref_fwdiffs = np.diff(ref_signal)[1:]
    ref_bwdiffs = np.diff(ref_signal)[:-1]

    for i in range(len(ref_fwdiffs)):
        ref_cur_sample = ref_signal[i + 1]
        ref_dist_fw = ref_fwdiffs[i]
        ref_dist_bw = ref_bwdiffs[i]

        dist_in_allowed_range = min_glottal_cycle <= ref_dist_fw <= max_glottal_cycle and min_glottal_cycle <= ref_dist_bw <= max_glottal_cycle
        if dist_in_allowed_range:

            cycle_start = ref_cur_sample - ref_dist_bw / 2
            cycle_stop = ref_cur_sample + ref_dist_fw / 2

            est_GCIs_in_cycle = est_signal[np.logical_and(est_signal > cycle_start, est_signal < cycle_stop)]
            n_est_in_cycle = np.count_nonzero(est_GCIs_in_cycle)

            nCycles += 1

            if n_est_in_cycle == 1:
                nHit += 1
                estimation_distance[nHit] = est_GCIs_in_cycle[0] - ref_cur_sample
            elif n_est_in_cycle < 1:
                nMiss += 1
            else:
                nFalse += 1

    estimation_distance = estimation_distance[np.invert(np.isnan(estimation_distance))]

    identification_rate = nHit / nCycles
    miss_rate = nMiss / nCycles
    false_alarm_rate = nFalse / nCycles
    identification_accuracy = 0 if np.size(estimation_distance) == 0 else np.std(estimation_distance)

    return {
        'identification_rate': identification_rate,
        'miss_rate': miss_rate,
        'false_alarm_rate': false_alarm_rate,
        'identification_accuracy': identification_accuracy
    }



def plot(egg,speech,gci,goi, trim = None, title = "Plot"):
    # window_start = 0
    # window_length = 16384
    
    gci_plot = np.zeros(len(egg))
    goi_plot = np.zeros(len(egg))

    gci_plot[gci] = 0.025
    goi_plot[goi] = -0.025

    plt.figure(figsize = (20, 12))
    
    # plt.plot(speech[window_start:window_start+window_length])
    plt.subplot(411)
    plt.plot(speech[:trim])
    plt.title('Speech signal', fontsize = 20)
    # plt.show()
    # plt.plot(egg[window_start:window_start+window_length])
    plt.subplot(412)
    plt.plot(egg[:trim])
    plt.title('EGG', fontsize = 20)
    # plt.show()
    # plt.plot(egg[window_start+1:window_start+window_length+1] - egg[window_start:window_start+window_length])
    plt.subplot(413)
    plt.plot(np.diff(egg)[:trim])
    plt.plot(gci_plot[:trim])
    plt.title('dEGG GCI', fontsize = 20)
    # plt.show()
    # plt.plot(gci_plot[window_start:window_start+window_length],label='gci')
    plt.subplot(414)
    plt.plot(np.diff(egg)[:trim])
    plt.plot(goi_plot[:trim])
    # plt.plot(goi_plot[window_start:window_start+window_length],label='goi')
    # plt.plot(goi_plot[:trim], label = 'goi')
    plt.title('dEGG GOI', fontsize = 20)
    # plt.legend()

    plt.subplots_adjust(top = 0.92)
    plt.suptitle(title, fontsize = 30)
    plt.show()

def main():
    pass
    
if __name__ == "__main__":
    main()