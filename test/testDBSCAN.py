#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np    

import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy.stats as stats
from itertools import combinations
import math
import os
import time


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
myhost = os.uname()[1]
print("Host:", myhost)

import cygno as cy
# import mylib as my
import ROOT
import root_numpy as rtnp

def Gauss3(x, a0, x0, s0):
    import numpy as np
    return a0 * np.exp(-(x - x0)**2 / (2 * s0**2))
    
def histGaus(var, bins, ax, color='b', xrange=False, alpha=1, label=''):
    from sklearn.metrics import r2_score
    from scipy.stats import chisquare
    import scipy.stats as stats
    if xrange:
        ax.hist(var, bins=bins, label=label, color=color, range=(xrange[0], xrange[1]), alpha=alpha)
        y, bins_edge = np.histogram(var, bins=bins, range=(xrange[0], xrange[1]))
    else:
        ax.hist(var, bins=bins, label=label, color=color)
        y, bins_edge = np.histogram(var, bins=bins)
    p0=[y.max(),bins_edge[y.argmax()], var.std()]
    x = np.linspace(bins_edge[0], bins_edge[-1], bins)
    popt, pcov = curve_fit(Gauss3, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    r2=r2_score(y, Gauss3(x, *popt))
    x = np.linspace(bins_edge[0], bins_edge[-1], 100)
    sstat='a = {0:.2f} +/- {1:.2f}\nb = {2:.2f} +/- {3:.2f} \
    \nc = {4:.2f} +/- {5:.2f}\n$R^2$ = {6:.4f}'.format(popt[0], perr[0], popt[1], perr[1], 
                                                popt[2], perr[2], r2)
    ax.plot(x,Gauss3(x, *popt), color+'--', label=sstat)
    return ax

# def pd_his_plot(what, ax, xmin=False, xmax=False, **key,):
#     if not xmin: xmin = what.min()
#     if not xmax: xmax = what.max()
#         
#     y = what[(what > xmin) & (what < xmax)]
# 
#     E = len(y)
#     M = y.mean()
#     S = y.std()
#     ax.set_title(what.name)
#     ax.hist(y, range=(xmin, xmax), **key, 
#             label='E = {:d}\nM = {:.2f}\nS = {:.2f}'.format(E, M, S))
#     ax.legend()
#     return ax

def pd_his_plot(what, ax, **key):
    E = len(what)
    M = what.mean()
    S = what.std()
    ax.set_title(what.name)
    ax.hist(what, label='E = {:d}\nM = {:.2f}\nS = {:.2f}'.format(E, M, S), **key)
    ax.legend()
    return ax


def ExpG(x, p0, p1, a0, x0, s0): # power of ten
    import numpy as np
    return p0*np.exp(p1*x) + a0 * np.exp(-(x - x0)**2 / (2 * s0**2))

def Exp2G(x, p0, p1, a0, x0, s0, a1, x1, s1): # power of ten
    import numpy as np
    return p0*np.exp(p1*x) + a0 * np.exp(-(x - x0)**2 / (2 * s0**2)) + a1 * np.exp(-(x - x1)**2 / (2 * s1**2))

def Exp(x, p0, p1): # power of ten
    import numpy as np
    return p0*np.exp(p1*x)

def G3(x, a0, x0, s0): # power of ten
    import numpy as np
    return a0 * np.exp(-(x - x0)**2 / (2 * s0**2))

def myExpG(data,ax, xmin, xmax, bins, E0, E1, a0, x0, s0):
    y = data[(data > xmin) & (data < xmax)]
    binsf = int(bins*((xmax-xmin)/(y.max()-y.min())))
    x=np.linspace(xmin , xmax, binsf)
    yh, _ = np.histogram(y, bins=binsf, range=(xmin,xmax))
    popt, pcov = curve_fit(ExpG, x, yh, 
                            p0=(E0, E1, a0, x0, s0))    
    #print ("Params: ", popt) 
    perr = np.sqrt(np.diag(pcov))
    #print ("err: ", perr)
    # plt.title(r'E0*$e^{(E1*x)}$ + a0*$e^{-\frac{(x - x0)^2}{2*s0^2}}$', fontsize=30)
    ax.plot(x, ExpG(x, *popt), 'r--', linewidth=2,
    label='E0 = %.2e +/- %.2e\nE1 = %.2e +/- %.2e\na0 = %.2e +/- %.2e\nx0 = %.2e +/- %.2e\ns0 = %.2e +/- %.2e' % 
                          (popt[0], perr[0],  popt[1],perr[1],
                           popt[2],perr[2], popt[3], perr[3], popt[4], perr[4]))
    ax.plot(x, G3(x, popt[2], popt[3], popt[4]), 'b-', label='signal')
    ax.plot(x, Exp(x, popt[0], popt[1]), 'y-', label='background')
    return ax, popt

def myExp2G(data,ax, xmin, xmax, bins, E0, E1, a0, x0, s0, a1, x1, s1):
    y = data[(data > xmin) & (data < xmax)]
    binsf = int(bins*((xmax-xmin)/(y.max()-y.min())))
    x=np.linspace(xmin , xmax, binsf)
    yh, _ = np.histogram(y, bins=binsf, range=(xmin,xmax))
    popt, pcov = curve_fit(Exp2G, x, yh, 
                            p0=(E0, E1, a0, x0, s0, a1, x1, s1))    
    #print ("Params: ", popt) 
    perr = np.sqrt(np.diag(pcov))
    #print ("err: ", perr)
    # plt.title(r'E0*$e^{(E1*x)}$ + a0*$e^{-\frac{(x - x0)^2}{2*s0^2}}$', fontsize=30)
    ax.plot(x, Exp2G(x, *popt), 'r--', linewidth=2,
    label='E0 = %.2e +/- %.2e\nE1 = %.2e +/- %.2e\na0 = %.2e +/- %.2e\nx0 = %.2e +/- %.2e\ns0 = %.2e +/- %.2e\
            \na1 = %.2e +/- %.2e\nx1 = %.2e +/- %.2e\ns1 = %.2e +/- %.2e ' % 
                          (popt[0], perr[0],  popt[1],perr[1],
                           popt[2],perr[2], popt[3], perr[3], popt[4], perr[4],
                           popt[5],perr[5], popt[6], perr[6], popt[7], perr[7]))
    ax.plot(x, G3(x, popt[2], popt[3], popt[4]), 'b-', label='signal')
    ax.plot(x, G3(x, popt[5], popt[6], popt[7]), 'k-', label='signal')
    ax.plot(x, Exp(x, popt[0], popt[1]), 'y-', label='background')
    return ax, popt


def n_std_rectangle(x, y, ax, image = np.array([]), n_std=3.0, facecolor='none', **kwargs):
    from matplotlib.patches import Rectangle
    mean_x = x.mean()
    mean_y = y.mean()
    std_x = x.std()
    std_y = y.std()
    half_width = n_std * std_x
    half_height = n_std * std_y
    if image.any():
        rimage = image*0
        xs = int(mean_x - half_width)+1
        xe = int(mean_x + half_width)+1
        ys = int(mean_y - half_height)+1
        ye = int(mean_y + half_height)+1
        # print(ys,ye, xs,xe)
        rimage[ys:ye, xs:xe]=image[ys:ye, xs:xe]
        # print (rimage)
        # print(rimage.sum())
        
    rectangle = Rectangle(
        (mean_x - half_width, mean_y - half_height),
        2 * half_width, 2 * half_height, facecolor=facecolor, **kwargs)
    return ax.add_patch(rectangle), rimage  

def confidence_ellipse(x, y, ax, image = np.array([]), n_std=3.0, facecolor='none', **kwargs):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    import numpy as np

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    if image.any():
        # ellsisse e' (x-x0)**2/a**2 + (y-y0)**2/b**2 < 1
        # print (mean_x, mean_y, ell_radius_x*scale_x, ell_radius_y*scale_y)
        rimage = image*0
        ar = abs(pearson)
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                xr = (y-mean_y)*np.sin(ar)+(x-mean_x)*np.cos(ar)
                yr = (y-mean_y)*np.cos(ar)-(x-mean_x)*np.sin(ar)
                if (xr)**2/(ell_radius_x*scale_x)**2 + (yr)**2/(ell_radius_y*scale_y)**2 < 1:
                    rimage[y,x]=image[y, x]
        # print (rimage)
        # print(rimage.sum())
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse), rimage 


def confidence_ellipse_par(x, y, image = np.array([]), n_std=3.0, facecolor='none', **kwargs):
    import numpy as np

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
                           
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
                           
    width=scale_x*ell_radius_x * 2 
    height=scale_y*ell_radius_y * 2              
    if image.any():
        # ellsisse e' (x-x0)**2/a**2 + (y-y0)**2/b**2 < 1
        # print (mean_x, mean_y, ell_radius_x*scale_x, ell_radius_y*scale_y)
        rimage = image*0
        ar = abs(pearson)
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                xr = (y-mean_y)*np.sin(ar)+(x-mean_x)*np.cos(ar)
                yr = (y-mean_y)*np.cos(ar)-(x-mean_x)*np.sin(ar)
                if (xr)**2/(ell_radius_x*scale_x)**2 + (yr)**2/(ell_radius_y*scale_y)**2 < 1:
                    rimage[y,x]=image[y, x]
        # print (rimage)
        # print(rimage.sum())
    else:
        rimage = np.array([])

    
    return width, height, pearson, rimage.sum(), np.size(rimage[rimage>0])

def proj_xy(img_data, x=True, y=True):
    import numpy as np
    import matplotlib.pyplot as plt
    if x:
        y_x = np.mean(img_data, axis=0)
        x_x = np.arange(0, img_data.shape[1])
        ax.plot(x_x, y_x, 'b', label='proj x')
    if y:
        y_y = np.mean(img_data, axis=1)
        x_y = np.arange(0, img_data.shape[0])
        ax.plot(x_y, y_y, 'r', label='proj y', alpha= 0.5)
    ax.legend()
    return ax


######################## INIT ###############################
# bandella
runI          = [4184, 4176, 4168, 4160, 4152, 4144, 4136, 4128, 4120] #440, 440, 440
runI          = [4185, 4177, 4169, 4161, 4153, 4145, 4137, 4129, 4121] #430, 440, 440
run_ped       = 4183
# KFC
#runI          = [4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047] #440, 440, 430
#run_ped       = 4038 
# # KFC cosmici
# runI          = [4035]
# run_ped       = 4034

cimax         = 500
cimin         = 0 
tag           = 'LAB'
nsigma        = 1.5 
fast          = False
doRescale     = False

version       = 'vF0'

###############################################################

cfile = cy.open_(runI[0], tag=tag, posix=False, verbose=False)
if cfile.x_resolution == 2048:
    rescale = 512 # orca flash
else:
    rescale = 576 # fusion


if not doRescale: 
    rescale = cfile.x_resolution
    eps=5
    min_samples = 40
    
    
else:
    eps=2
    min_samples = 5

        
tscale = int(cfile.x_resolution/rescale)
print("rascale: ", tscale)

################################################################
debug         = False
columns = ["iTr", "cluster_lable", "pixels", "photons", "ph_pixels", "x0start", "y0start", 
          "x0end", "y0end", "width", "height", "pearson"]


######################### Load PED #############################

m_image, s_image = cy.ped_(run_ped, path='./')

#########################

th_image   = np.round(m_image + nsigma*s_image)
print ("light over Th: %.2f " % (th_image.sum()-m_image.sum()))

start_time = time.time()
# loop sui fle da analizaare
for nRi in range(len(runI)):
    try:
        # open file
        cfile = cy.open_(runI[nRi], tag=tag, posix=False, verbose=True)
    except:
        print ('Problem in open file: ', runI[nRi])
        break
    #
    # crea un db vuoto da riempire 
    df = pd.DataFrame(columns = columns)
    # crea nome file di output
    file_out = ("dbscan_run%d_cmin_%d_cmax_%d_rescale_%d_nsigma_%.1f_ev_%d_ped_%d_%s.plk.gz" % 
                (runI[nRi], cimin, cimax, rescale, nsigma, cfile.max_pic, run_ped, version))
    #
######## main loop sulle imagini da analizzare ###########
    for iTr in range(0, cfile.max_pic): # cfile.max_image
        #if iTr % 10 == 0: 
          # running & debug ...
        print ('>>>> Processing RUN: ', runI[nRi], 'Event: ', iTr)
          # end 

        image = rtnp.hist2array(cfile.file.Get(cfile.pic[iTr])).T
        raw_image       = (image-m_image)
        rebin_image     = cy.rebin(raw_image, (rescale, rescale))  
        rebin_th_image  = cy.rebin((th_image-m_image), (rescale, rescale))
        edges           = (rebin_image > rebin_th_image) & (rebin_image < cimax)         
        points          = np.array(np.nonzero(edges)).T.astype(float) 
        
        # X_scaled        = StandardScaler().fit_transform(cy.poit_3d(points, rebin_image)) # 3D
        # dbscan          = DBSCAN(eps=0.045, min_samples = 20).fit(X_scaled) # eps=0.08 per 3D 0.04 2D
        
        dbscan          = DBSCAN(eps=eps, min_samples = min_samples).fit(points) 
        clusters        = dbscan.labels_
        n_points      = len(clusters>-1)
        if debug: print ("Clustering Elapsed time: {:.1f}".format(time.time() - start_time))
        for ic in range (0, max(clusters)+1):
            width = height = pearson = ph = dim = 0
            yc = points[:,0][clusters==ic]
            xc = points[:,1][clusters==ic]
            
            if fast:
                            #
            # attenzine NORMALIZZAZIONE per per rscale
            #
                ph, dim = cy.cluster_par(xc, yc, rebin_image) 
                width, height, pearson, _, _ = cy.confidence_ellipse_par(xc,yc)
                ph     = ph*tscale*tscale
                dim    = dim*tscale*tscale
                width  = width*tscale
                height = height*tscale            
                if debug: print ("Elapsed time fast: {:.1f}".format(time.time() - start_time))
            else:
                zx1 = int(xc.mean() - 5*xc.std())
                zx2 = int(xc.mean() + 5*xc.std())
                zy1 = int(yc.mean() - 5*yc.std())
                zy2 = int(yc.mean() + 5*yc.std())
                
                izoom = raw_image[zy1*tscale:zy2*tscale,zx1*tscale:zx2*tscale]            
                width, height, pearson, ph, dim = cy.confidence_ellipse_par((xc-zx1)*tscale, (yc-zy1)*tscale, 
                                                                            image = izoom)
                if debug: print ("Elapsed time slow: {:.1f}".format(time.time() - start_time))

            for j in range(0, xc.shape[0]):
                x=int(xc[j])
                y=int(yc[j])
                #ph += rebin_image[y,x]
                if j == 0:
                    x0start = x*tscale
                    y0start = y*tscale
            x0end = x*tscale
            y0end = y*tscale

            
            # 
            # salva info per ogni cluster
            #
            df = df.append({columns[0]:iTr, columns[1]:ic, columns[2]:dim, columns[3]:ph, columns[4]:ph/dim, 
                            columns[5]:x0start, columns[6]:y0start, columns[7]:x0end, columns[8]:y0end, 
                            columns[9]:width, columns[10]:height, columns[11]:pearson},
                            ignore_index=True)
########### Debug  #####################################
        if iTr % 10 == 0 or debug:
            print ("DEBUG: number of points, clusters: " +str(n_points), ic) 
            print ("Elapsed time 10 events: {:.1f}".format(time.time() - start_time))
            print ([str(columns[i])+': {:.2f}'.format(x) for i, x in enumerate (df.tail(1).values[0])])
            fig, ax = plt.subplots (1,2, figsize=(10,5))
            ax[0].imshow(rebin_image, vmin=-5, vmax=20)
            yc = points[:,0][clusters>-1]
            xc = points[:,1][clusters>-1]
            ax[0].plot(xc,yc, 'r.', markersize=1, label="ic"+str(ic))
            ax[1].imshow(edges, cmap='YlGnBu', vmin=0,vmax=1)
            plt.show()

            if ic>0:
                fig, ax = plt.subplots(1,5, figsize=(20,4))
                yc = points[:,0][clusters==ic]
                xc = points[:,1][clusters==ic]

                zx1 = int(xc.mean() - 5*xc.std())
                zx2 = int(xc.mean() + 5*xc.std())
                zy1 = int(yc.mean() - 5*yc.std())
                zy2 = int(yc.mean() + 5*yc.std())
                izoom = raw_image[zy1*tscale:zy2*tscale,zx1*tscale:zx2*tscale] 
                ax[0].imshow(izoom, vmin=-5, vmax=30, aspect="auto")
                
                py = np.sum(izoom, axis=0)
                px = np.sum(izoom, axis=1)
                
                x = np.linspace(0, py.size, py.size)
                ax[1].plot(x,py, "navy", label='Entries: {:d}\nMeans {:.1f}\nStd Dev {:.1f}'.format(x.size, 
                                                                                                    x.mean(), x.std()))
                x = np.linspace(0, px.size, px.size)
                ax[2].plot(x,px, "navy", label='Entries: {:d}\nMeans {:.1f}\nStd Dev {:.1f}'.format(x.size, 
                                                                                                    x.mean(), x.std()))

                ax[3].imshow(rebin_image[zy1:zy2,zx1:zx2], vmin=-5, vmax=30, aspect="auto")
                el_plt, el_par = cy.confidence_ellipse(xc-zx1, yc-zy1, ax[3], edgecolor='yellow')
                ax[3].scatter(xc-zx1, yc-zy1, color='red', label = ('(%.2f,%.2f)\nP:%.2f\nPh: %.2f\nS: %d (%.1f)' %
                                        (width, height, pearson, ph, dim, ph/dim)))
                ax[4].imshow(edges[zy1:zy2,zx1:zx2], aspect="auto")
                ax[1].legend()
                ax[2].legend()
                ax[3].legend()
                plt.show()
            start_time = time.time()
#################### close and save ################
    df.to_pickle(file_out, compression='gzip')
    print ("out file", file_out)