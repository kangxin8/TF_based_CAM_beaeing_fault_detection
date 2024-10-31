
from io import BytesIO
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal
import os
import torch
import scipy.io.wavfile
import matplotlib.pyplot as plt
from kymatio.torch import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from kymatio.datasets import fetch_fsdd
from plot_utils import *
from utils import *

def expand_array(arr, target_length):
   
    current_length = len(arr)
    if current_length == target_length:
        return arr
    else:
      
        repeat_factor = target_length // current_length
        remainder = target_length % current_length


        expanded_array = np.repeat(arr, repeat_factor)

       
        if remainder > 0:
            expanded_array = np.concatenate((expanded_array, arr[:remainder]))

        return expanded_array



class Sigmoid(object):
    
    def __init__(self, origin_yvalue=0.001, half_xvalue=1.0):
        self.origin_yvalue = origin_yvalue
        self.half_xvalue = half_xvalue

    def __call__(self, x):
        x = torch.tensor(x)
        sigma = math.exp(math.log(1 / self.origin_yvalue - 1) / self.half_xvalue) 
        sigma = torch.tensor(sigma)
        x = 1/(1 + torch.pow(sigma, (-x + self.half_xvalue)))  

        return x
# SCWT
class wavelet_scaterring_analysis:
   
    def __init__(self, J, Q):
        self.J = J
        self.Q = Q


    def standard_scattering(self, signal):
        T = signal.shape[-1]
        scattering = Scattering1D(self.J, T, self.Q, out_type='list')
        scattering_coefficients = scattering(signal)
        meta = scattering.meta()
       
        return scattering_coefficients, meta
  
    def no_lowpass_scattering(self, signal):
        T = 0
        scattering = Scattering1D(self.J, T, self.Q, out_type='list')
        scattering_coefficients = scattering(signal)
        meta = scattering.meta()
      
        return scattering_coefficients, meta
  
    def generate_filters(self, N):
        
        phi_f, psi1_f, psi2_f = scattering_filter_factory(N, self.J, self.Q, N)
        return phi_f, psi1_f, psi2_f


   
    def scattering_result_visualisation(self, signal, fs, lama1=None):
       
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1] 
        
        time_duration = (T-1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=True, out_type='array')  
        # scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list') 
        scattering_coefficients = scattering(signal) 
        meta = scattering.meta()  
        order0 = np.where(meta['order'] == 0) 
        order1 = np.where(meta['order'] == 1) 
        order2 = np.where(meta['order'] == 2)
        coefficients_order1 = scattering_coefficients[order1] 
        n_freq_order1 = meta['xi'][order1][:, 0] 
        freq_order1 = n_freq_order1 * fs 
        lama1 = n_freq_order1[lama1]
        index_psi1_filter = np.where(meta['xi'][:, 0] == lama1) 
        coefficients_order2 = scattering_coefficients[index_psi1_filter][1:] 
        n_freq_order2 = meta['xi'][index_psi1_filter][1:, 1]  
        freq_order2 = n_freq_order2 * fs  
        t = np.linspace(0, time_duration, scattering_coefficients.shape[-1])  
       
        fig1 = visuallize_2d_array(coefficients_order1, X=t, Y=freq_order1, method='p', title='Title') 
        fig2 = visuallize_2d_array(coefficients_order2, X=t, Y=freq_order2, method='p', title='Title') 

    def scattering_result_visualisation_for_CAM(self, signal, fs, non_threshod=95):
        
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1] 
        time_duration = (T-1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list')  
        scattering_coefficients = scattering(signal) 
        meta = scattering.meta() 
        # order0 = np.where(meta['order'] == 0) 
        order1 = np.where(meta['order'] == 1) 
        # order2 = np.where(meta['order'] == 2)
        
        coef_arr_list = [scattering_coefficients[int(index)]['coef'] for index in order1[0]]
        coefficients_order1 = [np.array(expand_array(arr, len(coef_arr_list[0]))) for arr in coef_arr_list]
        coefficients_order1 = np.array(coefficients_order1)
        
        sigma_order1 = [meta['sigma'][sigma_index, 0] for sigma_index in order1[0]] 
        norm_factor = np.array(sigma_order1)/sigma_order1[-1] 
        norm_coefficients_order1 = coefficients_order1 / norm_factor[:, np.newaxis]  
      
        Zxx = np.abs(norm_coefficients_order1)
        quartiles = np.percentile(Zxx.flatten(), [25, 50, non_threshod]) 
     
        origin_yvalue = 0.001
     
        half_xvalue = quartiles[2] * 100.0
        f = Sigmoid(origin_yvalue, half_xvalue) 
        ZXX_non = Zxx * 100.0  
        norm_nonlinearprocess_coefficients_order1 = f(ZXX_non) 
        n_freq_order1 = meta['xi'][order1][:, 0] 
        freq_order1 = n_freq_order1 * fs  
        t = np.linspace(0, time_duration, len(coef_arr_list[0]))  

       
        fig1, ax1 = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        pcm = ax1.pcolormesh(t, freq_order1, np.abs(norm_coefficients_order1), cmap='viridis', norm=None, vmin=None, vmax=None, shading=None, alpha=None)  #  'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ax1.axis('off')  
        plt.tight_layout(pad=0)  
        buf_original = BytesIO() 
        plt.savefig(buf_original, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1) 
        buf_original.seek(0)  
        original_img = Image.open(buf_original)
        # plt.imshow(original_img)
        # plt.axis('off') 
        # plt.show()

      
        fig2, ax2 = plt.subplots(1, 1, figsize=(3.2, 2.4), dpi=300)  # 双栏（3.2， 2.4）dpi=300
        pcm1 = ax2.pcolormesh(t, freq_order1, np.abs(norm_nonlinearprocess_coefficients_order1), cmap='viridis', norm=None, vmin=None, vmax=None, shading=None, alpha=None)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ax2.axis('off') 
        plt.tight_layout(pad=0) 
        buf_non = BytesIO()  
        plt.savefig(buf_non, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)  
        buf_non.seek(0) 
        processed_img = Image.open(buf_non)
        # plt.imshow(processed_img)
        # plt.axis('off') 
        # plt.show()

        return original_img, processed_img


    def no_lowpass_scattering_result_visualisation(self, signal, fs, lama1=None):
      
        signal = torch.from_numpy(signal).float()
        T = signal.shape[-1] 
        time_duration = (T-1) / fs
        scattering = Scattering1D(self.J, T, self.Q, average=False, out_type='list') 
        scattering_coefficients = scattering(signal) 
        meta = scattering.meta() 
        # order0 = np.where(meta['order'] == 0) 
        order1 = np.where(meta['order'] == 1)  
        # order2 = np.where(meta['order'] == 2)
       
        coef_arr_list = [scattering_coefficients[int(index)]['coef'] for index in order1[0]]
        coefficients_order1 = [np.array(expand_array(arr, len(coef_arr_list[0]))) for arr in coef_arr_list]
        coefficients_order1 = np.array(coefficients_order1)
    
        sigma_order1 = [meta['sigma'][sigma_index, 0] for sigma_index in order1[0]] 
        norm_factor = np.array(sigma_order1)/sigma_order1[-1]
        norm_coefficients_order1 = coefficients_order1 / norm_factor[:, np.newaxis] 
      
        Zxx = np.abs(norm_coefficients_order1)  
        quartiles = np.percentile(Zxx.flatten(), [25, 50, 95])  # 四分位数 ideal95
       
        origin_yvalue = 0.001
      
        half_xvalue = quartiles[2] * 100.0
        f = Sigmoid(origin_yvalue, half_xvalue) 
        ZXX_non = Zxx * 100.0  
        norm_nonlinearprocess_coefficients_order1 = f(ZXX_non) 
        n_freq_order1 = meta['xi'][order1][:, 0]  
        freq_order1 = n_freq_order1 * fs  
        # lama1 = n_freq_order1[lama1]
        # index_psi1_filter = np.where(meta['xi'][:, 0] == lama1) 
        # coefficients_order2 = scattering_coefficients[index_psi1_filter][1:] 
        # n_freq_order2 = meta['xi'][index_psi1_filter][1:, 1]  
        # freq_order2 = n_freq_order2 * fs 
        t = np.linspace(0, time_duration, len(coef_arr_list[0])) 
      
        fig1 = visuallize_2d_array(norm_coefficients_order1, X=t, Y=freq_order1, method='p', title='Title')
        fig2 = visuallize_2d_array(norm_nonlinearprocess_coefficients_order1 , X=t, Y=freq_order1, method='p', title='Title') 
        # fig2 = visuallize_2d_array(coefficients_order2, X=t, Y=freq_order2, method='p', title='Title')
        # resize_acc_cwt_coefs = resize(norm_coefficients_order1, output_shape=(224, 224),
        #                               anti_aliasing=True) 
        # acc_cwt_Fig_2d = visuallize_2d_array(resize_acc_cwt_coefs, X=t, Y=freq_order1, method='i', title='Title') 
   
        print('1')
        return fig1, fig2
  


