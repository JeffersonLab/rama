import numpy as np
import statistics as sts
from matplotlib import pyplot as plt
import time
import math

def constant_test(x):
    '''
    piecewise constant test function
    '''
    fun = ((x >= 0) * (x <= 1/2)) * 1 - ((x >= 1/2) * (x <= 1)) * 1
    return fun

def linear_test(x):
    '''
    piecewise linear test function
    '''
    fun = (1 - 4*x)*((x >= 0) * (x <= 1/2)) + (4*x - 3)* ((x >= 1/2) * (x <= 1))
    return fun

def sudden_change_data(mean1=5, mean2=3, sigma=0.5, change1=10000, change2=40000, total_time=50000) :

    np.random.seed(0)
    noise = np.random.normal(0, sigma, total_time)
    data = np.zeros((total_time, 1))
    data[:change1] = mean1
    data[change1:change2] = mean2
    data[change2:] = mean1
    data = data + noise.reshape((total_time, 1))

    return data

def gradual_change_data(mean1=8, mean2=2, sigma=0.05, start_change=100000, end_change=200000, total_time=300000) :

    np.random.seed(0)
    noise = np.random.normal(0, sigma, total_time)
    slope = (mean2-mean1)/(end_change-start_change)
    inter = mean1-slope*start_change
    data =  slope*np.arange(0, total_time) + inter
    data[:start_change] = mean1
    data[end_change:] = mean2
    data = data.reshape((total_time, 1)) + noise.reshape((total_time, 1))

    return data

def multiscalecoeff_seperatechange_online_shift(data, test, threshold, min_scale, max_scale, shiftscale):
    """
    Parameters
    ----------
    data : one dimenstion vector with lenghth n
    test : the detect function
    threshold : the threshold to conclusion when change happpens
    min_scale : the minimum scale, we add 2**min_scale at one time
    max_scale : the maximum scale, it is determined the detect accuracy
    shiftscale : shift scale

    Returns
    -------
    a : detect coeffiecint
    lower : the lower bound for change happens
    upper : the upper bound for change happens
    
    note : if the detect coefficient is less than the lower bound or greater 
            than the upper bound, then we conclude the change happens.
    """
    
    
    #--------------store data------
    stream = []
    anomaly = {}
    anomalyInteral = {}
    test_weight = {}
    coeff = {}           
    coeff_mean = {}                # store coeff mean
    coeff_variance = {}            # store coeff standard deviation
    coeff_upper = {}               # upper = mean + threshold * standard_dev
    coeff_lower = {}               # lower = mean - threshold * standard_dev

    
    for grade in range(min_scale, max_scale+1):
        gradename = 'level ' + str(grade)
        coeff[gradename] = []
        coeff_mean[gradename] = []
        coeff_variance[gradename] = []
        coeff_upper[gradename] = []
        coeff_lower[gradename] = []
        
        anomaly[gradename] = [0]
        anomalyInteral[gradename] = [[0, 0]]
        test_weight[gradename] = test( np.arange(0, 2**grade)/(2**grade) + 0.5/(2**grade)  )
    
        
        
    stream.extend(data[:2**max_scale])  # we initialy store 2^{max_scale} data

        
    i = 2**(max_scale - min_scale)
    
    
    while i < data.shape[0]//2**min_scale:
        
        # add new data
        stream.extend(data[i*(2**min_scale):(i+1)*(2**min_scale)])
        
        # compute coefficient for each garde
        for grade in range(min_scale, max_scale+1):
            
            gradename = 'level ' + str(grade)
            
            # apply shift technique
            for j in range(0, 2**min_scale, 2**shiftscale):   
                temp_sidx = i*(2**min_scale)-2**grade + j
                temp_eidx = i*(2**min_scale) + j
                tempdata = stream[temp_sidx:temp_eidx]
                temp_test_weight = test_weight[gradename]
                temp_coeff = temp_test_weight@tempdata / (2**(2*grade))      # this is equation (5) in current draft
                coeff[gradename].append( temp_coeff )
                
                if len(coeff_mean[gradename])==0:
                    last_mean = temp_coeff
                    last_variance = 0
                    temp_mean = temp_coeff
                    temp_variance = 0
                    temp_upper = temp_coeff
                    temp_lower = temp_coeff
                else:
                    last_mean = coeff_mean[gradename][-1]
                    last_variance = coeff_variance[gradename][-1]
                    len_tempcoeff = len(coeff[gradename])
                    temp_mean = last_mean + (temp_coeff - last_mean) / len_tempcoeff    # this is equation (14) in current draft
                    temp_variance = (len_tempcoeff-2)/(len_tempcoeff-1) * last_variance + (temp_coeff - last_mean) ** 2 / len_tempcoeff # this is equation (15) in current draft
                    
                    #print(temp_variance, temp_mean)
                    
                    temp_upper = temp_mean + threshold * np.sqrt( temp_variance )
                    temp_lower = temp_mean - threshold * np.sqrt( temp_variance )
                    
                
                coeff_upper[gradename].append( temp_upper )
                coeff_lower[gradename].append(  temp_lower ) 
                
                if temp_coeff >= temp_lower and temp_coeff <= temp_upper:
                    coeff_mean[gradename].append( temp_mean )
                    coeff_variance[gradename].append(  temp_variance )
                else:
                    coeff_mean[gradename].append( last_mean )
                    coeff_variance[gradename].append(  last_variance )  
                    
                
                    
                if temp_coeff < temp_lower or temp_coeff > temp_upper:
                    last_sidx = anomalyInteral[gradename][-1][0]
                    last_sidx = anomalyInteral[gradename][-1][1]
                    if temp_sidx > last_sidx:
                        print('ANOMALY: scale: {}, partition: [{},{}], amplitude: {}, lower: {}, upper: {}'.format(grade, temp_sidx, temp_eidx, temp_coeff, temp_lower, temp_upper))
                        anomaly[gradename].append(len(coeff[gradename]))
                        anomalyInteral[gradename].append([temp_sidx, temp_eidx])
            
        i += 1
        
    return coeff, coeff_upper, coeff_lower, anomalyInteral, anomaly


def plot_results(data, coeff, coeff_upper, coeff_lower, anomalyInteral, min_scale, max_scale, changeType, testfunctionType):
    
    for grade in range(min_scale, max_scale+1):
        
        gradename = 'level ' + str(grade)
        plt.figure(2*grade-1)
        plt.plot(range(len(coeff_lower[gradename])), np.array(coeff_lower[gradename]), 'tab:red')
        plt.plot(range(len(coeff_upper[gradename])), np.array(coeff_upper[gradename]), 'tab:red')
        plt.scatter(range(len(coeff[gradename])), coeff[gradename], linewidths=0.5)
        plt.xlabel('buckets')
        plt.ylabel('coefficeint')
        plt.title('scale {}'.format(grade))
        
        fig_filename = 'Fig/{}/{}_{}_grade{}_coeff.png'.format(changeType, changeType, testfunctionType, grade)
        plt.savefig(fig_filename)        
        
        plt.show()
        
        
        tempchangeInterval = anomalyInteral[gradename]
        plt.figure(2*grade)
        plt.scatter(range(0, data.shape[0]), data)

        
        for changeInterval in tempchangeInterval:
            plt.scatter(range(changeInterval[0], changeInterval[1]), data[changeInterval[0]:changeInterval[1]], color='tab:red')
            
        plt.xlabel('time')
        plt.ylabel('value')
        
        
        fig_filename = 'Fig/{}/{}_{}_grade{}_origial.png'.format(changeType, changeType, testfunctionType, grade)
               
        plt.title('scale {}'.format(grade))
            
        plt.savefig(fig_filename)
        plt.show()
        
        

    return 
  


