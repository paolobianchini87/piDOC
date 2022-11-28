''' Series of functions used to:
1) predict mass, age and distance of a series of simulated clusters observations (test set) using pi-DOC:
function: make_prediction()
2) plot the figures comparing true expected values and predicted values
functions: display_mosaic_prediction(), display_predicted_vs_true_mass(), display_predicted_vs_true_total_mass(), display_predicted_vs_true_age(), pdf_distances()
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from tensorflow.keras.models import *
import time


def make_predictions(pathtrained,pathinput,pathprediction,L_testset):
    '''Make the mass predictions using the convolutional encoder-decoder (CED) and distance and age prediction using the CNN network.
    The input are the flux maps of the test set (L_testset)
    pathtrained : path of the trained models (encoder-decoder CED.json and CNN network CNN.json)
    pathinput: path of the input maps
    pathprediction : path of the directory in which to save the predictions
    L_testset : flux maps of the test set that are to be used as input of pi-DOC
    '''
    
# Mass predictions from Convolutional Encoder-Decoer (CED) of pi-DOC
    #name of the trained algorithm
    name_CED='CED_200.h5'
    # load json and create model
    json_file  =  open(pathtrained+'CED.json', 'r')
    loaded_model_json  =  json_file.read()
    json_file.close()
    trained_network  =  model_from_json(loaded_model_json)
    # load weights into new model
    trained_network.load_weights(pathtrained+name_CED) #this is the trained model

    #put the input in standardized form (the algorithm was trained with a standardized data set; see Section 3 of C&B2021 )
    mean_L=np.load(pathinput+'standardization_files/meanluminosityVtrain.npy') #mean to be used, issued from the training set
    std_L=np.load(pathinput+'standardization_files/stdluminosityVtrain.npy') #standard deviation to be used, issued from the training set
    L=L_testset
    L=(L-mean_L)
    L=L/std_L
    maps_tot=np.shape(L)[0]
    print('\n****************************************************')
    print('*******  compute predictions with pi-DOC  ********\n')
    print('total number of maps used as input:', maps_tot)
    
    #make mass predictions for all the training set, giving (standardized) luminsoity maps as input
    start_CED_prediction = time.time()

    print('- - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('              start mass predictions              ')
    print('- - - - - - - - - - - - - - - - - - - - - - - - \n')
    prediction  =  trained_network.predict(L)
    #put the mass prediction in original (de-standardized) form
    mean_M=np.load(pathinput+'standardization_files/meanmasstrain.npy') #mean to be used, issued from the training set
    std_M=np.load(pathinput+'standardization_files/stdmasstrain.npy') #standard deviation to be used, issued from the training set
    prediction=(prediction*std_M)+mean_M
    
    print('--> Mass predictions done for {} maps using pi-DOC Convolutional Encoder-Decoder'.format(maps_tot))
    # save predictions in a npy file
    np.save(pathprediction+'predictionM.npy',prediction)
    print('file "'+pathprediction+'predictionM.npy" saved')
    print('CED prediction computation time per map : {:.3f} s'.format((time.time()-start_CED_prediction)/maps_tot)+'\n')

# Distance and age predictions from Convolutional Neural Network of pi-DOC
    # load json and create model
    #name of the trained algorithm
    name_CNN='CNN_100.h5'
    json_file  =  open(pathtrained+'CNN.json', 'r')
    loaded_model_json  =  json_file.read()
    json_file.close()
    trained_network  =  model_from_json(loaded_model_json)
    # load weights into new model
    trained_network.load_weights(pathtrained+name_CNN) #this is the trained model

    #make distance and age predictions for all the training set, giving (standardized) luminsoity maps as input
    start_CNN_prediction = time.time()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('       start distance and age predictions         ')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - \n')
    prediction_distance_age  =  trained_network.predict(L)
    #put the age and distance predictions in original (de-standardized) form
    mean_d_age=np.load(pathinput+'standardization_files/mean_distance_age_train.npy')  #mean to be used, issued from the training set
    std_d_age=np.load(pathinput+'standardization_files/std_distance_age_train.npy') #standard deviation to be used, issued from the training set
    prediction_distance_age = (prediction_distance_age*std_d_age)+mean_d_age
   
    print('--> Distance and age predictions done for {} maps using pi-DOC Convolutional Neural Network'.format(maps_tot))
    # save predictions in a npy file
    np.save(pathprediction+'predictiondistance_age.npy',prediction_distance_age)
    print('file "'+pathprediction+'predictiondistance_age.npy" saved')
    print('CNN prediction computation time : {:.3f} s'.format((time.time()-start_CNN_prediction)/maps_tot)+'\n')

    print('***********  end pi-DOC predictions  ***************')
    print('****************************************************\n')
    return

def display_mosaic_prediction(pathimage,L_testset,M_testset,distanceage_testset,prediction_M,prediction_distance_age):
    '''Displays a mosaic of 4 maps next to each other, the first row being the flux mass (used as input of pi-DOC), the second row is the true mass map and the third row the predicted mass map. True and predicted age/distance are plotted as labels.
    pathimage: location of the saved figure with the name of "mosaic_predictions_maps.png"
    L_testset: input flux maps
    M_testset: true mass maps of the input luminosity maps
    distanceage_testset: true distances and age of the input luminosity maps
    prediction_M: mass maps predicted by pi-DOC
    prediction_distance_age: ange and distances predicted by pi-DOC
    '''
    title_font  =  {'size':'7','color':'white'}
    distance_font = {'size':'5.5','color':'yellow'}
    age_font = {'size':'5.5','color':'red'}
    map_label_font = {'size':'7','color':'white'}
    
    #select randomly 4 maps for which you will plot the predictions
    n = 4
    maps_tot=np.shape(M_testset)[0]
    start_maps = random.sample(range(0, maps_tot), n)

    fig,ax  =  plt.subplots(nrows = 3, ncols = n,sharex = 'col', sharey = 'row')
    map_label = ['a','b','c','d']
   
    for j in range(n):
        #plot the 4 selected luminosity maps
        im1 = ax[0,j].imshow(L_testset[start_maps[j],:,:,0],extent = [-20,20,-20,20],cmap = 'bone',vmin = -6,vmax = -3)
        ax[0,0].text(-10,17,'Input flux',title_font)
        ax[0,0].set_ylabel('y (arcsec)')
        plt.xticks([-20,-10,0,10,20])
        ax[0,j].text(-17,-18,'({})'.format(map_label[j]),map_label_font)
       
        #plot the 4 corresponding true mass maps + labels of their true distance and age values
        im2 = ax[1,j].imshow(M_testset[start_maps[j],:,:,0],extent = [-20,20,-20,20],cmap = 'gist_heat',vmin = -1.5,vmax = 0.75)
        t = ax[1,j].text(-17,-18,'D = {:.1f} kpc'.format(distanceage_testset[start_maps[j]][0]),distance_font)
        t.set_bbox(dict(facecolor = 'black', alpha = 0.5,edgecolor = 'none'))
        t = ax[1,j].text(3,-18,'A = {:.1f} Gyr'.format(distanceage_testset[start_maps[j]][1]),age_font)
        t.set_bbox(dict(facecolor = 'black', alpha = 0.5,edgecolor = 'none'))
        ax[1,0].text(-10,17,'Real mass',title_font)
        ax[1,0].set_ylabel('y (arcsec)')
       
        #plot the 4 corresponding predicted mass maps from pi-DOC + labels with their predicted distance and age values
        im3 = ax[2,j].imshow(prediction_M[start_maps[j],:,:,0],extent = [-20,20,-20,20],cmap = 'gist_heat',vmin = -1.5,vmax = 0.75)
        ax[2,0].text(-13,17,'Predicted mass',title_font)
        ax[2,0].set_ylabel('y (arcsec)')
        ax[2,j].set_xlabel('x (arcsec)')
        t = ax[2,j].text(-17,-18,'D = {:.1f} kpc'.format(prediction_distance_age[start_maps[j]][0]),distance_font)
        t.set_bbox(dict(facecolor = 'black', alpha = 0.5,edgecolor = 'none'))
        t = ax[2,j].text(3,-18,'A = {:.1f} Gyr'.format(prediction_distance_age[start_maps[j]][1]),age_font)
        t.set_bbox(dict(facecolor = 'black', alpha = 0.5,edgecolor = 'none'))
        
        ax[2,0].set_xticks([-20,-10,0,10,20])
        ax[2,0].set_xticklabels([-20,-10,0,10,20])
        if(j>0):
            ax[0,j].tick_params(axis = 'y',which = 'both',left = False)
            ax[1,j].tick_params(axis = 'y',which = 'both',left = False)
            ax[2,j].tick_params(axis = 'y',which = 'both',left = False)
            ax[2,j].set_xticks([-10,0,10,20])
            ax[2,j].set_xticklabels([-10,0,10,20])

    colorbar(im1,'$log_{10}(F/L_{\\odot}.kpc^{2})$')
    colorbar(im2,'$log_{10}(M/M_{\\odot})$')
    colorbar(im3,'$log_{10}(M/M_{\\odot})$')
    plt.subplots_adjust(wspace = 0,hspace = 0)
    plt.suptitle('Top: V-band flux maps \n Middle: true mass maps, distance and age \n Bottom: predicted mass maps, distance and age',fontsize=10)
    plt.savefig(pathimage+'mosaic_predictions_maps_{}-{}-{}-{}.png'.format(start_maps[0],start_maps[1],start_maps[2],start_maps[3]),dpi=300)
    
    print('--> Figure of 4 mass maps, with distance and age predictions saved: '+pathimage+'mosaic_predictions_maps_{}-{}-{}-{}.png'.format(start_maps[0],start_maps[1],start_maps[2],start_maps[3]))
    return


def display_predicted_vs_true_mass(pathimage,M_testset,prediction_M):
    '''Displays the predicted vs true graph (2D histogram) of the pixel-by-pixel mass using the all the maps of the test set
    pathimage: location of the resulting figure saved under the name of 'Mass_map_true_vs_predicted.png'
    M_testset: the true mass maps of the test set sample
    prediction_M: the predicted mass maps from pi-DOC
    This routine also computes statistics of the performances and saves them in tables as .txt files
    '''
    plt.clf()

    #define the bins and the limits of the 2D histogram
    nbrbin = 50
    lim1 = -2
    lim2 = 1
    binx = biny = np.linspace(lim1,lim2,nbrbin)
        
    listM = np.reshape(M_testset[:,:,:,0],(np.shape(M_testset)[0]*np.shape(M_testset)[1]**2))
    listprediction = np.reshape(prediction_M[:,:,:,0],(np.shape(prediction_M)[0]*np.shape(prediction_M)[1]**2))

    fig = plt.figure(0)

    # make the histogram
    histo = np.histogram2d(listM,listprediction, bins = (binx, biny))
    H = histo[0].T
    xedges = histo[1]
    yedges = histo[2]

    xcenter = (xedges[1:]+xedges[:-1])/2
    
    #calculate the statistics
    meanrx,stdrx,percenterrorx = np.zeros(len(H)),np.zeros(len(H)),np.zeros(len(H))
    for i in range(len(H)):
        test = np.logical_and(listM < xedges[i+1],listM >=  xedges[i])
        tmp = listprediction[test]
        meanrx[i] = np.mean(tmp-xcenter[i])
        stdrx[i] = np.std(tmp-xcenter[i])
        abserr = np.abs(np.power(10,tmp)-np.power(10,xcenter[i]))/np.power(10,tmp)
        percenterrorx[i] = np.sum(100*abserr)/len(tmp)

    ycenter = (yedges[1:]+yedges[:-1])/2

    meanry,stdry,percenterrory = np.zeros(len(H)),np.zeros(len(H)),np.zeros(len(H))
    for i in range(len(H)):
        test = np.logical_and(listprediction < yedges[i+1],listprediction >=  yedges[i])
        tmp = listM[test]
        meanry[i] = np.mean(tmp-ycenter[i])
        stdry[i] = np.std(tmp-ycenter[i])
        abserr = np.abs(np.power(10,tmp)-np.power(10,ycenter[i]))/np.power(10,tmp)
        percenterrory[i] = np.sum(100*abserr)/len(tmp)

    # Statistics
    meanerrory_all = np.nanmean(percenterrory)
    minerrory_all = np.nanmin(percenterrory)
    maxerrory_all = np.nanmax(percenterrory)
    closest = closest_in_list(biny,-1)[0]
    lowlim = np.where(biny == closest)[0][0]
    meanerrory_withoutlowbins = np.nanmean(percenterrory[lowlim:]) #low mass (log10(M) < -1) excluded
    minerrory_withoutlowbins = np.nanmin(percenterrory[lowlim:])
    maxerrory_withoutlowbins = np.nanmax(percenterrory[lowlim:])

    # Statistics format __._ %
    meanerrory_all = float("{:.1f}".format(meanerrory_all))
    minerrory_all = float("{:.1f}".format(minerrory_all))
    maxerrory_all = float("{:.1f}".format(maxerrory_all))

    print('Mass mean percentage error : {} %'.format(meanerrory_all))

    # Saving stastics tables in a .txt file
    table = [['Avg % error','Min % error','Max % error'],[meanerrory_all,minerrory_all,maxerrory_all]]
    with open(pathimage+'predicted_vs_true_mass.txt', 'w') as f:
        f.write(tabulate(table,headers = 'firstrow'))

    # Display the figure
    X,Y = np.meshgrid(xedges,yedges)
    gs  =  gridspec.GridSpec(2, 3, width_ratios = [1,3,0.25], height_ratios = [3,1])
    ax1  =  plt.subplot(gs[0,1])
    ax2  =  plt.subplot(gs[0,0])
    ax3  =  plt.subplot(gs[1,1])
    axecolbar  =  plt.subplot(gs[0,2])
    im = ax1.pcolormesh(X,Y,H,norm = LogNorm())
    cbar = fig.colorbar(im,cax = axecolbar)
    cbar.set_label('Counts',rotation = 90)
    axecolbar.tick_params(axis = 'y',which = 'minor',right = False)
    ax1.plot([lim1,lim2],[lim1,lim2],'-r')  # one-to-one relation
    ax1.set_xlim([lim1,lim2])
    ax1.set_ylim([lim1,lim2])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    ax2.errorbar(meanry,ycenter,xerr = stdry) #vertical graph
    ax2.plot([0,0],[lim1,lim2],'r')
    ax2.set_xlim([-0.5,0.5])
    ax2.set_ylim([lim1,lim2])
    ax2.set_xlabel(r"$\mathrm{\mu}$")
    ax2.set_ylabel('Predicted $log_{10}(M/M_{\odot})$')

    ax3.errorbar(xcenter,meanrx,stdrx) #horizontal graph
    ax3.plot([lim1,lim2],[0,0],'r')
    ax3.set_xlim([lim1,lim2])
    ax3.set_ylim([-0.5,0.5])
    ax3.set_xlabel('True $log_{10}(M/M_{\odot})$')
    ax3.set_ylabel(r"$\mathrm{\xi}$")

    plt.suptitle('Predicted vs true mass distribution')
    plt.tight_layout()
    plt.subplots_adjust(top = 0.88,wspace = 0.4, hspace = 0.2)
    fig.savefig(pathimage+'Mass_map_true_vs_predicted.png', bbox_inches  =  "tight")

    print('--> Figure pixel-by-pixel predicted mass vs. true mass saved')
    return

def display_predicted_vs_true_total_mass(pathimage,M_testset,prediction_M):
    '''Displays the predicted vs true graph (2D histogram) of the total mass using the all the maps of the test set and predictions
    pathimage: the location of the resulting figure saved under the name of 'Total_mass_map_true_vs_predicted.png'
    M_testset: the true mass maps of the test set sample
    prediction_M: the predicted mass maps from pi-DOC
    This also computes statistics of the performances and saves them in tables as .txt files.
    '''
    #transform mass values in linear scale
    M_testset = np.power(10,M_testset)
    prediction_M = np.power(10,prediction_M)
    #calculate total mass of a given map of the test set
    M_testset = np.reshape(M_testset[:,:,:,0],(np.shape(M_testset)[0],np.shape(M_testset)[1]**2))
    list_tm_true = np.sum(M_testset,axis = 1)
    #calculate total mass of a given predicted mass map
    prediction_M = np.reshape(prediction_M[:,:,:,0],(np.shape(prediction_M)[0],np.shape(prediction_M)[1]**2))
    list_tm_predicted = np.sum(prediction_M,axis = 1)
    #take the logarithm of the total mass
    list_tm_true = np.log10(list_tm_true)
    list_tm_predicted = np.log10(list_tm_predicted)

    #define the histogram
    lim1 = 3
    lim2 = 4.5
    fig = plt.figure(2)
    nbrbin = 20
    binx = biny = np.linspace(lim1,lim2,nbrbin)
    histo = np.histogram2d(list_tm_true,list_tm_predicted,bins = (binx, biny))
    H = histo[0].T
    xedges = histo[1]
    yedges = histo[2]

    xcenter = (xedges[1:]+xedges[:-1])/2

    meanrx,stdrx,percenterrorx = np.zeros(len(H)),np.zeros(len(H)),np.zeros(len(H))
    for i in range(len(H)):
        test = np.logical_and(list_tm_true < xedges[i+1],list_tm_true >=  xedges[i])
        tmp = list_tm_predicted[test]
        meanrx[i] = np.mean(tmp-xcenter[i])
        stdrx[i] = np.std(tmp-xcenter[i])
        abserr = np.abs(np.power(10,tmp)-np.power(10,xcenter[i]))/np.power(10,tmp)
        percenterrorx[i] = np.sum(100*abserr)/len(tmp)

    ycenter = (yedges[1:]+yedges[:-1])/2
    meanry,stdry,percenterrory = np.zeros(len(H)),np.zeros(len(H)),np.zeros(len(H))
    for i in range(len(H)):
        test = np.logical_and(list_tm_predicted < yedges[i+1],list_tm_predicted >=  yedges[i])
        tmp = list_tm_true[test]
        meanry[i] = np.mean(tmp-ycenter[i])
        stdry[i] = np.std(tmp-ycenter[i])
        abserr = np.abs(np.power(10,tmp)-np.power(10,ycenter[i]))/np.power(10,tmp)
        percenterrory[i] = np.sum(100*abserr)/len(tmp)

    # Statistics
    meanerrory_all = np.nanmean(percenterrory)
    minerrory_all = np.nanmin(percenterrory)
    maxerrory_all = np.nanmax(percenterrory)

    # Statistics format __._ %
    meanerrory_all = float("{:.1f}".format(meanerrory_all))
    minerrory_all = float("{:.1f}".format(minerrory_all))
    maxerrory_all = float("{:.1f}".format(maxerrory_all))

    print('Total mass mean percentage error : {} %'.format(meanerrory_all))

    # Saving stastics tables in a .txt file
    table = [['Average % error','Minimum % error','Maximum % error'],[meanerrory_all,minerrory_all,maxerrory_all]]
    with open(pathimage+'predicted_vs_true_total_mass.txt', 'w') as f:
        f.write(tabulate(table,headers = 'firstrow'))

    # Display the figure
    X,Y = np.meshgrid(xedges,yedges)
    gs  =  gridspec.GridSpec(2, 3, width_ratios = [1,3,0.25], height_ratios = [3,1])
    ax1  =  plt.subplot(gs[0,1])
    ax2  =  plt.subplot(gs[0,0])
    ax3  =  plt.subplot(gs[1,1])
    axecolbar  =  plt.subplot(gs[0,2])
    im = ax1.pcolormesh(X,Y,H,norm = LogNorm())
    cbar = fig.colorbar(im,cax = axecolbar)
    cbar.set_label('Counts',rotation = 90)
    axecolbar.tick_params(axis = 'y',which = 'minor',right = False)
    ax1.plot([lim1,lim2],[lim1,lim2],'-r')  # one-to-one relation
    ax1.set_xlim([lim1,lim2])
    ax1.set_ylim([lim1,lim2])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    ax2.errorbar(meanry,ycenter,xerr = stdry) #vertical graph
    ax2.plot([0,0],[lim1,lim2],'r')
    ax2.set_xlim([-0.5,0.5])
    ax2.set_ylim([lim1,lim2])
    ax2.set_xlabel(r"$\mathrm{\mu}$")
    ax2.set_ylabel('Predicted $log_{10}(M_{tot}/M_{\odot})$')

    ax3.errorbar(xcenter,meanrx,stdrx) #horizontal graph
    ax3.plot([lim1,lim2],[0,0],'r')
    ax3.set_xlim([lim1,lim2])
    ax3.set_ylim([-0.5,0.5])
    ax3.set_xlabel('True $log_{10}(M_{tot}/M_{\odot})$')
    ax3.set_ylabel(r"$\mathrm{\xi}$")

    plt.suptitle('Predicted vs true total mass')
    plt.tight_layout()
    plt.subplots_adjust(top = 0.88,wspace = 0.4, hspace = 0.2)
    fig.savefig(pathimage+'Total_mass_map_true_vs_predicted.png')

    print('--> Figure predicted total mass vs. true total mass saved')
    return

def display_predicted_vs_true_age(pathimage,distanceage_testset,distanceage_prediction):
    '''Displays the predicted vs true graph (2D histogram) of the age using all maps of the test set and corresponding predictions.
    pathimage: the location of the resulting figure saved under the name of 'Age_true_vs_predicted.png'
    distanceage_testset: the true distance and ages of the test set sample
    distanceage_prediction: the predicted distances and ages from pi-DOC
    This also computes statistics of the performances and saves them in tables as .txt files
    '''
    plt.clf()
    
    age_testset=distanceage_testset[:,1]  #select only the age of the testset
    age_prediction=distanceage_prediction[:,1] #select only the age of the predictions

    #define the bins and the limits of the 2D histogram
    nbrbin = 20
    start=0 #Gyr
    lim=14 #Gyr
    binx = biny = np.linspace(start,lim,nbrbin)

    fig = plt.figure(0)

    # make the histogram
    histo = np.histogram2d(age_testset,age_prediction, bins = (binx, biny))
    H = histo[0].T
    xedges = histo[1]
    yedges = histo[2]

    xcenter = (xedges[1:]+xedges[:-1])/2

    #calculate the statistics
    meanrx,stdrx,abserrorx = np.zeros(len(H)),np.zeros(len(H)),np.zeros(len(H))
    for i in range(len(H)):
        test = np.logical_and(age_testset < xedges[i+1],age_testset >=  xedges[i])
        tmp = age_prediction[test]
        meanrx[i] = np.mean(tmp-xcenter[i])
        stdrx[i] = np.std(tmp-xcenter[i])
        abserr = np.abs(tmp-xcenter[i])
        abserrorx[i] = np.sum(abserr)/len(tmp)

    ycenter = (yedges[1:]+yedges[:-1])/2
    dy = yedges[1]-yedges[0]
    meanry,stdry,abserrory = np.zeros(len(H)),np.zeros(len(H)),np.zeros(len(H))
    for i in range(len(H)):
        test = np.logical_and(age_prediction < yedges[i+1],age_prediction >=  yedges[i])
        tmp = age_testset[test]
        meanry[i] = np.mean(tmp-ycenter[i])
        stdry[i] = np.std(tmp-ycenter[i])
        abserr = np.abs(tmp-ycenter[i])
        abserrory[i] = np.sum(abserr)/len(tmp)

    # Statistics
    meanerrory_all = np.nanmean(abserrory)
    minerrory_all = np.nanmin(abserrory)
    maxerrory_all = np.nanmax(abserrory)

    # Statistics format __._ %
    meanerrory_all = float("{:.2f}".format(meanerrory_all))
    minerrory_all = float("{:.2f}".format(minerrory_all))
    maxerrory_all = float("{:.2f}".format(maxerrory_all))

    print('Mean age absolute error : {} Gyr'.format(meanerrory_all))

    # Saving stastics tables in a .txt file
    table = [['Mean absolute error (Gyr)','Minimum absolute error (Gyr)','Maximum absolute error (Gyr)'],[meanerrory_all,minerrory_all,maxerrory_all]]
    with open(pathimage+'predicted_vs_true_age.txt', 'w') as f:
        f.write(tabulate(table,headers = 'firstrow'))

    # Display the figure
    X,Y = np.meshgrid(xedges,yedges)
    gs  =  gridspec.GridSpec(2, 3, width_ratios = [1,3,0.25], height_ratios = [3,1])
    ax1  =  plt.subplot(gs[0,1])
    ax2  =  plt.subplot(gs[0,0])
    ax3  =  plt.subplot(gs[1,1])
    axecolbar  =  plt.subplot(gs[0,2])
    im = ax1.pcolormesh(X,Y,H,norm = LogNorm())
    cbar = fig.colorbar(im,cax = axecolbar)
    cbar.set_label('Counts',rotation = 90)
    axecolbar.tick_params(axis = 'y',which = 'minor',right = False)
    ax1.plot([start,lim],[start,lim],'-r')  # one-to-one relation
    ax1.set_xlim([start,lim])
    ax1.set_ylim([start,lim])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    ax2.errorbar(meanry,ycenter,xerr = stdry) #vertical graph
    ax2.plot([0,0],[start,lim],'r')
    ax2.set_xlim([-5,5])
    ax2.set_ylim([start,lim])
    ax2.set_xlabel(r"$\mathrm{\mu}$")
    ax2.set_ylabel('Predicted age (Gyr)')
    
    ax3.errorbar(xcenter,meanrx,stdrx) #horizontal graph
    ax3.plot([start,lim],[0,0],'r')
    ax3.set_xlim([start,lim])
    ax3.set_ylim([-5,5])
    ax3.set_xlabel('True age (Gyr)')
    ax3.set_ylabel(r"$\mathrm{\xi}$")

    plt.suptitle('Predicted vs true age')
    plt.tight_layout()
    plt.subplots_adjust(top = 0.88,wspace = 0.4, hspace = 0.2)
    fig.savefig(pathimage+'Age_true_vs_predicted.png')

    print('--> Figure predicted age vs true age saved')
    return

def pdf_distances(pathimage,distanceage_testset,distanceage_prediction):
    '''For each expected distances in the test set (15,30,60,80 kpc), this function selects all the associated predicted maps and plots the histogram of the predicted distances probability distributions functions.
    pathimage: the location of the resulting figure saved under the name of 'Age_true_vs_predicted.png'
    distanceage_testset: the true distance and ages of the test set sample
    distanceage_prediction: the predicted distances and ages from pi-DOC
    This also creates a .txt file with a table containing the statistics of the distributions.
    '''

    distance_testset=distanceage_testset[:,0]  #select only the distance of the testset
    distance_prediction=distanceage_prediction[:,0] #select only the distance of the predictions
    
    # divide the predictions according their corresponding real distances (15, 30, 60, 80 kpc)
    indices15 = np.where(distance_testset == 15.)
    predist15 = distance_prediction[indices15]
    indices30 = np.where(distance_testset == 30.)
    predist30 = distance_prediction[indices30]
    indices60 = np.where(distance_testset == 60.)
    predist60 = distance_prediction[indices60]
    indices80 = np.where(distance_testset == 80.)
    predist80 = distance_prediction[indices80]

    # define the bins for the histogram
    range15 = np.linspace(np.min(distance_prediction[indices15]),np.max(distance_prediction[indices15]),int(len(predist15)/3))
    range30 = np.linspace(np.min(distance_prediction[indices30]),np.max(distance_prediction[indices30]),int(len(predist30)/3))
    range60 = np.linspace(np.min(distance_prediction[indices60]),np.max(distance_prediction[indices60]),int(len(predist60)/3))
    range80 = np.linspace(np.min(distance_prediction[indices80]),np.max(distance_prediction[indices80]),int(len(predist80)/3))

    # compute the histogram
    pdf15,bins15 = np.histogram(predist15,bins = range15,density = True)
    pdf30,bins30 = np.histogram(predist30,bins = range30,density = True)
    pdf60,bins60 = np.histogram(predist60,bins = range60,density = True)
    pdf80,bins80 = np.histogram(predist80,bins = range80,density = True)
    
    # compute the statistics of the distributions
    meanpredicted15 = np.mean(predist15)
    meanpredicted30 = np.mean(predist30)
    meanpredicted60 = np.mean(predist60)
    meanpredicted80 = np.mean(predist80)
    stdpredicted15 = np.std(predist15)
    stdpredicted30 = np.std(predist30)
    stdpredicted60 = np.std(predist60)
    stdpredicted80 = np.std(predist80)
    meanstd = np.mean([stdpredicted15,stdpredicted30,stdpredicted60,stdpredicted80])
    meanabserrorpredicted15 = np.mean(np.abs(predist15-distance_testset[indices15]))
    meanabserrorpredicted30 = np.mean(np.abs(predist30-distance_testset[indices30]))
    meanabserrorpredicted60 = np.mean(np.abs(predist60-distance_testset[indices60]))
    meanabserrorpredicted80 = np.mean(np.abs(predist80-distance_testset[indices80]))
    meanerr = np.mean([meanabserrorpredicted15,meanabserrorpredicted30,meanabserrorpredicted60,meanabserrorpredicted80])

    print('Average absolute error on four distances : {:.2f} kpc'.format(meanerr))

    # Format and print the statistics
    meanpredicted15 = float("{:.2f}".format(meanpredicted15))
    meanpredicted30 = float("{:.2f}".format(meanpredicted30))
    meanpredicted60 = float("{:.2f}".format(meanpredicted60))
    meanpredicted80 = float("{:.2f}".format(meanpredicted80))
    stdpredicted15 = float("{:.2f}".format(stdpredicted15))
    stdpredicted30 = float("{:.2f}".format(stdpredicted30))
    stdpredicted60 = float("{:.2f}".format(stdpredicted60))
    stdpredicted80 = float("{:.2f}".format(stdpredicted80))
    meanstd = float("{:.2f}".format(meanstd))
    meanabserrorpredicted15 = float("{:.2f}".format(meanabserrorpredicted15))
    meanabserrorpredicted30 = float("{:.2f}".format(meanabserrorpredicted30))
    meanabserrorpredicted60 = float("{:.2f}".format(meanabserrorpredicted60))
    meanabserrorpredicted80 = float("{:.2f}".format(meanabserrorpredicted80))
    meanerr = float("{:.2f}".format(meanerr))

    table_with_avg = [['Real distance (kpc)',15,30,60,80,'Avg'],['Mean predicted distance (kpc)',meanpredicted15,meanpredicted30,meanpredicted60,meanpredicted80,'/'],['Standard deviation (kpc)',stdpredicted15,stdpredicted30,stdpredicted60,stdpredicted80,meanstd],['Mean absolute error (kpc)',meanabserrorpredicted15,meanabserrorpredicted30,meanabserrorpredicted60,meanabserrorpredicted80,meanerr]]
    with open(pathimage+'distance_predictions_stats.txt', 'w') as f:
        f.write(tabulate(table_with_avg))

    # Plot the histogram
    plt.clf()
    lim = 0.2
    plt.stairs(pdf15,bins15,fill=True,alpha=0.5)
    plt.plot([15,15],[0,lim],'--k',color='blue')
    plt.stairs(pdf30,bins30,fill=True,alpha=0.5)
    plt.plot([30,30],[0,lim],'--k',color='orange')
    plt.stairs(pdf60,bins60,fill=True,alpha=0.5)
    plt.plot([60,60],[0,lim],'--k',color='green')
    plt.stairs(pdf80,bins80,fill=True,alpha=0.5)
    plt.plot([80,80],[0,lim],'--k',color='red')
    plt.xlabel('Distance (kpc)')
    plt.ylabel('PDF')
    plt.xlim(0,100)
    plt.ylim(0,lim)
    plt.tight_layout()
    plt.savefig(pathimage+'Pdf_distances.png')
    print('--> PDF distances figure saved')
    return


def colorbar(mappable,label = None):
    ''' Creates a colourbar attached to a mappable object (e.g. mappable = image = plt.imshow(...)) with the label given in argument.
    Avoids issues that arise with subplots and colourbars.
    '''
    last_axes  =  plt.gca()
    ax  =  mappable.axes
    fig  =  ax.figure
    divider  =  make_axes_locatable(ax)
    cax  =  divider.append_axes("right", size = "5%", pad = -0.05)
    cbar  =  fig.colorbar(mappable, cax = cax)
    cbar.set_label(label,rotation = 90)
    plt.sca(last_axes)
    return cbar

def closest_in_list(myList, myNumber):
    """
    Takes a sorted list and a number as input and returns the closest value to this number in the list.If two numbers are equally close, return the smallest number. In this version, also returns the index.
    """
    myList = np.asarray(myList)
    idx = (np.abs(myList - myNumber)).argmin()
    return(myList[idx],idx)

def moving_average(a,n):
    '''Takes an array and a window width (total width, not half width) and returns the smoothed array, using the unchanged values of the input array for extreme values. If n = 0 then the function returns the original array.
    '''
    if(n == 0):
        return(a)
    ret  =  np.cumsum(a,dtype = float)
    ret[n:] = ret[n:]-ret[:-n]
    smtl = ret[n-1:]/n
    smtlf = np.append([],a[0:n//2])
    smtlf = np.append(smtlf,smtl)
    smtlf = np.append(smtlf,a[-n//2+1:])
    return(smtlf)
