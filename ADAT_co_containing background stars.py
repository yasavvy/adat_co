import os
import numpy as np
import pandas as pd
from astropy.io import fits
import hdbscan
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_cluster_with_isochrone(ax, cluster_mask, data, iso_data, coord_pair, cluster_color, label):
    ax.scatter(data[coord_pair[0]][~cluster_mask], data[coord_pair[1]][~cluster_mask], color='gray', label='Other Stars', s=5, alpha=0.7)
    ax.scatter(data[coord_pair[0]][cluster_mask], data[coord_pair[1]][cluster_mask], color='red', label='Cluster Stars', s=5, alpha=0.7)

    ax.plot(iso_data['G_BPmag'], iso_data['Gmag'],
            color='darkblue', linestyle='dashed', label='Isochrone', linewidth=2)

    ax.set_xlabel(coord_pair[0])
    ax.set_ylabel(coord_pair[1])
    ax.legend()


#Load your fits data
fits_file_path = '/your path'  #Use your path, I uploaded Gaia data from my drive
hdulist = fits.open(fits_file_path)
data = hdulist[1].data
hdulist.close()

#Load isochrones (you are welcome to use my file "Isochrones" in csv)
iso_file_path = '/your path.csv'  #Use your path, I uploaded Gaia data from my drive
iso_data = pd.read_csv(iso_file_path)

#Coordinates for clustering
selected_cols = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity']
X = np.array([data.field(col) for col in selected_cols]).T

#HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = clusterer.fit_predict(X)

#Create a dataframe to store cluster information
df_clusters = pd.DataFrame({'label': labels})

#Clusters selection conditions
min_cluster_size = 100

#Create a directory for saving images
output_directory = 'Your path'
os.makedirs(output_directory, exist_ok=True)

#Validation and saving images
for cluster_label, cluster_size in df_clusters['label'].value_counts().items():
    if cluster_size >= min_cluster_size and cluster_label != -1:
        cluster_mask = (labels == cluster_label)

        valid_mask = np.isfinite(data['phot_bp_mean_mag']) & np.isfinite(data['phot_rp_mean_mag'])
        valid_cluster_mask = np.isin(np.arange(len(data)), np.nonzero(cluster_mask)[0]) & valid_mask

        if np.sum(valid_cluster_mask) >= min_cluster_size:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            coord_pairs = [('dec', 'ra'), ('pmdec', 'pmra'), ('radial_velocity', 'parallax')]

            for i, coord_pair in enumerate(coord_pairs):
                plot_cluster_with_isochrone(axes[i], cluster_mask, data, iso_data, coord_pair=coord_pair,
                                            cluster_color=plt.cm.jet(cluster_label / len(df_clusters['label'].unique())), label=f'Cluster {cluster_label}')

            #Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=df_clusters['label'].min(), vmax=df_clusters['label'].max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes, orientation='vertical', fraction=0.03, pad=0.1)
            cbar.set_label('Cluster Label')

            #Save the image (I used tiff)
            output_filename = f'{output_directory}cluster_{cluster_label}_isochrone.tif'
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            
            plt.close()

print(f'Your results landed here: {output_directory}')
