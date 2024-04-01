import matplotlib as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from pod.regression import fit_linear_model_mle
from math import isnan, isinf
from scipy.stats import norm

def plot_linear_fit(ax,x,y):
    coef = fit_linear_model_mle(x, y, y_th = 0)
    
    #x_reg = np.asarray([np.min(x),np.max(x)])
    y_reg = x*coef['slope'] + coef['intercept']
    ax.plot(x,y_reg,)
     
    # tau == sigma
    # e = y - y_reg
    # coef['tau'] = np.sqrt(np.sum(e**2)/len(x))

    str = [f'{k}: {v:.3f}' for k,v in coef.items()]
    at = AnchoredText('\n'.join(str), prop=dict(size=15), frameon=False, loc='center left')
    ax.add_artist(at)

    return coef 
    
def plot_pod_estimate(ax,x,y,coef):
    threshold=17;
    
    var_0   = len(x)/coef['sigma']**2
    var_1   = sum(x**2)/coef['sigma']**2
    cov_para= sum(x)/coef['sigma']**2
    var_2   = 2*len(x)/coef['sigma']**2
    FIM     = np.array([[var_0   , cov_para, 0    ],
                        [cov_para, var_1   , 0    ],
                        [0       , 0       , var_2]])
    
    cov1    = np.linalg.inv(FIM)
    
    threshold = 0
    mu      = (threshold-coef['intercept']) / coef['slope']
    sigma   = coef['sigma']/coef['slope']
    if sigma <= 0:
        print('Sigma is negative \nMultiply with -1 tp proceed')
        sigma = sigma*-1
        
    phi     = np.array([[1 , 0    ],
                        [mu, sigma],
                        [0 , -1   ]])/coef['slope']

    pcov=np.dot(np.dot(np.transpose(phi),cov1),phi)

    # Calculating t_50, t_90 and t_90_95
    t_50    =norm.ppf(0.5,mu,sigma)
    t_90    =norm.ppf(0.9,mu,sigma)
    print(f'mu: {mu}\nsigma: {sigma}\nnorm.ppf(0.9,mu,sigma): {norm.ppf(0.9,mu,sigma)}')
    wp      =norm.ppf(0.9,0,1)
    sd      =np.sqrt(pcov[0,0]+wp*wp*pcov[1,1]+2*wp*pcov[0,1])
    print(f't_90: {t_90}\nwp: {wp}\nsd: {sd}')
    t_90_95 =t_90+1.645*sd
    print(t_90_95)



def format_axis(ax,xlabel,ylabel):
    plt.style.use('classic')
    ax.grid('on')
    ax.tick_params(axis='x', labelsize= 13)
    ax.tick_params(axis='y', labelsize= 13)

    ax.set_xlabel(xlabel = xlabel, fontsize= 15)
    ax.set_ylabel(ylabel = ylabel, fontsize= 15)
    return

def plot_pod(ax,x,y,xlabel,ylabel):
    ax.scatter(x,y, s = 1,alpha=1,color = 'b', label='data')    

    format_axis(ax,xlabel,ylabel)
    coef = plot_linear_fit(ax,x,y)
    plot_pod_estimate(ax,x,y,coef)
    coef['tf_x'] = 'x'
    coef['tf_y'] = 'y'
    return coef

def plot_pod_logx(ax,x_,y_,xlabel,ylabel):
    x_ = -x_ + 50
    str = f'x = np.log10(-x + 50)'
    at = AnchoredText(str, prop=dict(size=15), frameon=False, loc='lower center')
    ax.add_artist(at)

    # remove potential zeros
    #x = np.asarray([v[0] for v in zip(x_,y_) if not (v[0] == 0 or isinf(v[0]) or isnan(v[0]))])
    #y = np.asarray([v[1] for v in zip(x_,y_) if not (v[0] == 0 or isinf(v[0]) or isnan(v[0]))])
    
    x = np.log10(x_)
    y = y_
    
    xlabel = 'log( ' + xlabel + ' )'
    ax.scatter(x,y, s = 1,alpha=1,color = 'b', label='data')    

    format_axis(ax,xlabel,ylabel)
    
    coef = plot_linear_fit(ax,x,y)
    coef['tf_x'] = str
    coef['tf_y'] = 'y'
    return coef

def plot_pod_logy(ax,x_,y_,xlabel,ylabel):
    print(f'{type(x_)}, {type(y_)}')
    print(f'{len(x_)}, {len(y_)}')

    #x = np.asarray([v[0] for v in zip(x_,y_) if not (v[1] == 0 or isinf(v[1]) or isnan(v[1]))])
    #y = np.asarray([v[1] for v in zip(x_,y_) if not (v[1] == 0 or isinf(v[1]) or isnan(v[1]))])
    x = x_
    y = np.log10(y_)

    print(f'{type(x)}, {type(y)}')
    print(f'{len(x)}, {len(y)}')

    ylabel = 'log( ' + ylabel + ' )'

    ax.scatter(x,y, s = 1,alpha=1,color = 'b', label='data')    

    format_axis(ax,xlabel,ylabel)

    coef = plot_linear_fit(ax,x,y)
    coef['tf_x'] = f'x'
    coef['tf_y'] = 'log10(y)'
    return coef

def plot_pod_logx_logy(ax,x_,y_,xlabel,ylabel):
    x_ = -x_ + 50
    str = f'x = np.log10(-x + 50)'
    at = AnchoredText(str, prop=dict(size=15), frameon=False, loc='lower center')
    ax.add_artist(at)

    #x = np.asarray([v[0] for v in zip(x_,y_) if not (v[0] == 0 or isinf(v[0]) or isnan(v[0]))])
    #y = np.asarray([v[1] for v in zip(x_,y_) if not (v[0] == 0 or isinf(v[0]) or isnan(v[0]))])

    #x_ = np.asarray([v[0] for v in zip(x,y) if not (v[1] == 0 or isinf(v[1]) or isnan(v[1]))])
    #y_ = np.asarray([v[1] for v in zip(x,y) if not (v[1] == 0 or isinf(v[1]) or isnan(v[1]))])

    x = np.log10(x_)
    y = np.log10(y_)

    xlabel = 'log( ' + xlabel + ' )'
    ylabel = 'log( ' + ylabel + ' )'

    ax.scatter(x,y, s = 1,alpha=1,color = 'b', label='data')    
    
    format_axis(ax,xlabel,ylabel)

    coef = plot_linear_fit(ax,x,y)
    coef['tf_x'] = str
    coef['tf_y'] = 'log10(y)'
    return coef