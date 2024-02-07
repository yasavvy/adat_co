import os
import numpy as np
import pandas as pd
from astropy.io import fits
import hdbscan
import matplotlib.pyplot as plt
from isochrones import get_ichrone
from astroML.plotting import scatter_contour

#Load your fits data
fits_file_path = '/your path' #Use your path, I uploaded Gaia data from my drive
hdulist = fits.open(fits_file_path)
data = hdulist[1].data
hdulist.close()

#Select coordinates for clustering
selected_cols = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity']
X = np.array([data.field(col) for col in selected_cols]).T

#Apply HDBSCAN algorithm for clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5) #Please, edit if required
labels = clusterer.fit_predict(X)

#Create a dataframe to store cluster information
df_clusters = pd.DataFrame({'label': labels})

#Set conditions for selecting clusters
min_cluster_size = 100

#Create a directory for saving images
output_directory = '/your path/' #Please edit
os.makedirs(output_directory, exist_ok=True)

#Photometric validation and saving images
for cluster_label, cluster_size in df_clusters['label'].value_counts().items():
    if cluster_size >= min_cluster_size:
        #Filter for the current cluster
        cluster_mask = (labels == cluster_label)

        #Photometric validation
        valid_mask = np.isfinite(data['bp_rp']) & np.isfinite(data['phot_g_mean_mag'])
        valid_cluster_mask = np.isin(np.arange(len(data)), np.nonzero(cluster_mask)[0]) & valid_mask

        if np.sum(valid_cluster_mask) >= min_cluster_size:
            #Plots for the current cluster
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            #Plot coordinates ra and dec
            scatter_contour(data['ra'][valid_cluster_mask], data['dec'][valid_cluster_mask], ax=axes[0],
                            filled=True, color='red', label=f'Cluster {cluster_label}')
            axes[0].set_title(f'HDBSCAN Clustering - Cluster {cluster_label}')
            axes[0].set_xlabel('Right Ascension (ra)')
            axes[0].set_ylabel('Declination (dec)')
            axes[0].legend()

            #Choose pair for the second plot
            second_coord_pair = ('pmra', 'pmdec')

            #Plot in the selected coordinate pair
            scatter_contour(data[second_coord_pair[0]][valid_cluster_mask],
                            data[second_coord_pair[1]][valid_cluster_mask], ax=axes[1],
                            filled=True, color='red', label=f'Cluster {cluster_label}')
            axes[1].set_title(f'Cluster Distribution in {second_coord_pair[0]} and {second_coord_pair[1]} - Cluster {cluster_label}')
            axes[1].set_xlabel(second_coord_pair[0])
            axes[1].set_ylabel(second_coord_pair[1])
            axes[1].legend()

            #Choose pair for the third plot
            third_coord_pair = ('parallax', 'radial_velocity')

            #Plot in the selected coordinate pair
            scatter_contour(data[third_coord_pair[0]][valid_cluster_mask],
                            data[third_coord_pair[1]][valid_cluster_mask], ax=axes[2],
                            filled=True, color='red', label=f'Cluster {cluster_label}')
            axes[2].set_title(f'Cluster Distribution in {third_coord_pair[0]} and {third_coord_pair[1]} - Cluster {cluster_label}')
            axes[2].set_xlabel(third_coord_pair[0])
            axes[2].set_ylabel(third_coord_pair[1])
            axes[2].legend()

            #Isochrone from the library
            isochrone = get_ichrone('mist')

            #Add isochrone to the plots
            isochrone_plot_args = dict(color='blue', linestyle='dashed', label='Isochrone Approximation')

            #Isochrone for coordinates ra and dec
            axes[0].plot(isochrone['G_BP'] - isochrone['G_RP'] + isochrone['G'],
                         isochrone['G'], **isochrone_plot_args)
            axes[0].legend()

            #Isochrone for coordinates pmra and pmdec
            axes[1].plot(isochrone['pmra'], isochrone['pmdec'], **isochrone_plot_args)
            axes[1].legend()

            #Isochrone for coordinates parallax and radial_velocity
            axes[2].plot(isochrone['parallax'], isochrone['radial_velocity'], **isochrone_plot_args)
            axes[2].legend()

            #Save the image (I used tiff)
            output_filename = f'{output_directory}cluster_{cluster_label}_isochrone.tif'
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            
            plt.close()

print(f'Your results landed here: {output_directory}')
