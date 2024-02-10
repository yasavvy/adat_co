import os
import numpy as np
import pandas as pd
from astropy.io import fits
import hdbscan
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cluster_with_isochrone(ax, cluster_mask, data, iso_data, coord_pair, cluster_color, label):
    ax.scatter(data[coord_pair[0]][~cluster_mask], data[coord_pair[1]][~cluster_mask], color='gray', label='Other Stars', s=5, alpha=0.7)
    ax.scatter(data[coord_pair[0]][cluster_mask], data[coord_pair[1]][cluster_mask], color='red', label='Cluster Stars', s=5, alpha=0.7)

    ax.plot(iso_data['G_BPmag'], iso_data['Gmag'], color='darkblue', linestyle='dashed', label='Isochrone', linewidth=2)

    ax.set_xlabel(coord_pair[0])
    ax.set_ylabel(coord_pair[1])
    ax.legend()

def plot_cluster_2d(ax, cluster_mask, data, cluster_color, label):
    ax.scatter(data['phot_bp_mean_mag'][~cluster_mask], data['phot_rp_mean_mag'][~cluster_mask], color='gray', label='Other Stars', s=5, alpha=0.7)
    ax.scatter(data['phot_bp_mean_mag'][cluster_mask], data['phot_rp_mean_mag'][cluster_mask], color='red', label='Cluster Stars', s=5, alpha=0.7)

    ax.set_xlabel('G_BPmag')
    ax.set_ylabel('G_RPmag')
    ax.legend()
    ax.set_title(f'Cluster {label}')

#Load your fits data
fits_file_path = 'Your path' #Use your path, I uploaded Gaia data from my drive
hdulist = fits.open(fits_file_path)
data = hdulist[1].data
hdulist.close()

#Load isochrones (you are welcome to use my files containing Isochrones)
iso_file_path = '/your path.csv'  #Use your path, I uploaded data from my drive
iso_data = pd.read_csv(iso_file_path)

#Coordinates for clustering
selected_cols = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'phot_g_mean_mag']
X = np.array([data.field(col) for col in selected_cols]).T

#HDBSCAN for clastering
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = clusterer.fit_predict(X)

#Create a dataframe to store cluster information
df_clusters = pd.DataFrame({'label': labels})

#Clasters selection conditions
min_cluster_size = 100

#Create a directory for saving images
output_directory = 'your path'
os.makedirs(output_directory, exist_ok=True)

#Validation and saving images
for cluster_label, cluster_size in df_clusters['label'].value_counts().items():
    if cluster_size >= min_cluster_size and cluster_label != -1:
        cluster_mask = (labels == cluster_label)

        valid_mask = np.isfinite(data['phot_bp_mean_mag']) & np.isfinite(data['phot_rp_mean_mag'])
        valid_cluster_mask = np.isin(np.arange(len(data)), np.nonzero(cluster_mask)[0]) & valid_mask

        if np.sum(valid_cluster_mask) >= min_cluster_size:
            #2D image after clastering
            fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
            plot_cluster_2d(ax_2d, cluster_mask, data,
                            cluster_color=plt.cm.jet(cluster_label / len(df_clusters['label'].unique())), label=f'Cluster {cluster_label}')

            #Saving image in TIFF
            output_filename_2d = f'{output_directory}cluster_{cluster_label}_2d.tif'
            plt.savefig(output_filename_2d, bbox_inches='tight', dpi=300)
            plt.close(fig_2d)

            #3D image with isochrone - validation
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')

            plot_cluster_with_isochrone(ax_3d, cluster_mask, data, iso_data,
                                        coord_pair=('phot_bp_mean_mag', 'phot_g_mean_mag'),
                                        cluster_color=plt.cm.jet(cluster_label / len(df_clusters['label'].unique())), label=f'Cluster {cluster_label}')

            #Adding a colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=df_clusters['label'].min(), vmax=df_clusters['label'].max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_3d, orientation='vertical', fraction=0.03, pad=0.1)
            cbar.set_label('Cluster Label')

            #Saving 3D image in TIFF
            output_filename_3d = f'{output_directory}cluster_{cluster_label}_isochrone_3d.tif'
            plt.savefig(output_filename_3d, bbox_inches='tight', dpi=300)
            plt.close(fig_3d)

print(f'Your results landed here: {output_directory}')
