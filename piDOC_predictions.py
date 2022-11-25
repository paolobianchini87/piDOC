'''ALGORITHM π-DOC - Version 0.1
The algorithm is presented in: Chardin & Bianchini 2021 (C&B21), MNRAS, 504, 5656
"Predicting images for the dynamics of stellar clusters (π-DOC): a deep learning framework to predict mass, distance, and age of globular clusters"
Contributions to the code:
Paolo Bianchini, December '22
Vincent Duret, July '21
Jonathan Chardin & Paolo Bianchini,  April '21

Description:
The deep learning algorithm π-DOC predicts the dynamical mass distribution of a globular cluster, its age and its distance, taking as a input a V-band flux map of a cluster.
This python code demonstrates the application of π-DOC on a set of mock V-band observations of globular clusters, subsample of the test set described in C&B21, Section 2.3.
It uses the following functions (from functions.py):
    --> Application of the algorithm π-DOC on 100 input flux maps:
    (1) make_predictions(), to predict mass map, age and distance from a cluster flux map
    (2) display_mosaic_prediction(), to plot the predicted properties vs. the real properties for 4 randomly selected cases
    --> Plot and compute the performances of π-DOC on the entire sample of 100 input maps:
    (3) display_predicted_vs_true_mass(), to plot the pixel-by-pixel predicted mass values vs, the expected ones
    (4) display_predicted_vs_true_total_mass(), to plot the predicted total mass values vs. the expected ones
    (5) display_predicted_vs_true_age(), to plot the predicted ages vs. the expected ones
    (6) pdf_distances(), to plot the predicted distances vs. the expected ones
Inputs:
    input_files/luminosity_V_testset.npy: 100 maps of V-band flux [ log10(F/(L_sun/kpc^2)), F is the flux value]; maps size: 40 arcsec X 40 arcsec (160 px X 160 px); pixel scale: 0.25 arcsec
    input_files/mass_testset.npy: true mass maps for the 100 flux maps above [ log10(M/M_sun), M is the mass]; same map size and pixel scale as above
    input_files/distance_age_testset.npy: ture distance in kpc and age in Gyr for each of the 100 maps above
Outputs:
in predictions/:
    --> Predictions of π-DOC:
    "predictionM.npy" and "predictiondistance_age.npy": predcited mass maps and predicated distance and age (same format as input files)
    --> Figures:
     - mosaic_predictions_maps.png
     - Mass_map_true_vs_predicted.png
     - Total_mass_map_true_vs_predicted.png
     - Age_true_vs_predicted.png
     - Pdf_distances.png
    --> Tables (statistics of the performance of π-DOC):
     - predicted_vs_true_mass.txt
     - predicted_vs_true_total_mass.txt
     - predicted_vs_true_age.txt
     - distance_predictions_stats.txt
Run: python3 piDOC_prediction.py
'''


import numpy as np
import matplotlib
import os
from functions import *

### MAIN ##

pathtestset = 'input_files/'  #location of input files
pathtrained = 'trained_piDOC/'  #location of the trained π-DOC algorithm
pathprediction = 'predictions/' #location of output files (predictions and figures)
try:
    os.mkdir(pathprediction)
except OSError as error:
    print(error)

''' ---> FIRST PART: make predicitons and plot the results'''
''' Apply π-DOC to a set of flux maps to predict the corresponding dynamical mass map, distance and age '''
# input luminosity maps, used as input of the algorithm π-DOC (subsample of the test set used in C&B21, see sect. 2.3)
L_testset=np.load(pathtestset+'luminosity_V_testset.npy')
# true expected mass maps, ages and distances of the luminosity maps (these are the quantities that π-DOC will predict and that we will use to quantify the performances of the algorithm)
M_testset=np.load(pathtestset+'mass_testset.npy')
distanceage_testset=np.load(pathtestset+'distance_age_testset.npy')

# this is the trained algorithm π-DOC, it predict the mass map, age and distance of given input luminosity maps
# it saves the mass predictions and the distance/age predictions in the files: pathprediction/predictiondistance_age.npy and pathprediction/predictiondistance_age.npy
make_predictions(pathtrained,pathtestset,pathprediction,L_testset)

''' Plot the results '''
# open files with mass predicitons and distance/age predictions
prediction_M=np.load(pathprediction+'predictionM.npy')
prediction_distanceage=np.load(pathprediction+'predictiondistance_age.npy')

# plot 4 maps (chosen randomly from the test set sample) and their corresponding predicitons (equivalebt of Fig. 4 of C&B21)
display_mosaic_prediction(pathprediction,L_testset,M_testset,distanceage_testset,prediction_M,prediction_distanceage)

''' ---> SECOND PART:'''
''' Additional figures to evaluate the performaces of the π-DOC on a large sample of input maps '''
# plot pixel-by-pixel mass predicitons vs the true expected values (equivalent of Fig. 5 of C&B21)
display_predicted_vs_true_mass(pathprediction,M_testset,prediction_M)
# plot total mass predicitons vs the true expected values (equivalent of Fig. 6 of C&B21)
display_predicted_vs_true_total_mass(pathprediction,M_testset,prediction_M)
# plot age predicitons vs the true expected ages (equivalent of Fig. 8 of C&B21)
display_predicted_vs_true_age(pathprediction,distanceage_testset,prediction_distanceage)
# plot the distribution of predictd distances (equivalent to Fig. 9 of C&B21)
pdf_distances(pathprediction,distanceage_testset,prediction_distanceage)
