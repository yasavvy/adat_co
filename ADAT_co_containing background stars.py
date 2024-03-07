import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
import hdbscan

class Plotter:
    def __init__(self, ax, data, cluster_mask, cluster_color, label):
        self.ax = ax
        self.data = data
        self.cluster_mask = cluster_mask
        self.cluster_color = cluster_color
        self.label = label

    def plot(self):
        raise NotImplementedError

class IsochronePlotter(Plotter):
    def __init__(self, ax, data, cluster_mask, cluster_color, label, iso_data):
        super().__init__(ax, data, cluster_mask, cluster_color, label)
        self.iso_data = iso_data

    def plot(self):
        self.ax.scatter(self.data['phot_bp_mean_mag'][~self.cluster_mask], self.data['phot_g_mean_mag'][~self.cluster_mask], color='gray', label='Other Stars', s=5, alpha=0.7)
        self.ax.scatter(self.data['phot_bp_mean_mag'][self.cluster_mask], self.data['phot_g_mean_mag'][self.cluster_mask], color='red', label='Cluster Stars', s=5, alpha=0.7)
        self.ax.plot(self.iso_data['G_BPmag'], self.iso_data['Gmag'], color='darkblue', linestyle='dashed', label='Isochrone', linewidth=2)
        self.ax.set_xlabel('G_BPmag')
        self.ax.set_ylabel('Gmag')
        self.ax.legend()

class ClusterPlotter(Plotter):
    def plot(self):
        self.ax.scatter(self.data['phot_bp_mean_mag'][~self.cluster_mask], self.data['phot_rp_mean_mag'][~self.cluster_mask], color='gray', label='Other Stars', s=5, alpha=0.7)
        self.ax.scatter(self.data['phot_bp_mean_mag'][self.cluster_mask], self.data['phot_rp_mean_mag'][self.cluster_mask], color='red', label='Cluster Stars', s=5, alpha=0.7)
        self.ax.set_xlabel('G_BPmag')
        self.ax.set_ylabel('G_RPmag')
        self.ax.legend()
        self.ax.set_title(f'Cluster {self.label}')

class ClusterAnalyzer:
    def __init__(self, fits_file_path, iso_file_path, output_directory, min_cluster_size=100):
        self.fits_file_path = fits_file_path
        self.iso_file_path = iso_file_path
        self.output_directory = output_directory
        self.min_cluster_size = min_cluster_size

    def run_analysis(self):
        hdulist = fits.open(self.fits_file_path)
        data = hdulist[1].data
        hdulist.close()

        iso_data = pd.read_csv(self.iso_file_path)

        selected_cols = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'radial_velocity', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'phot_g_mean_mag']
        X = np.array([data.field(col) for col in selected_cols]).T

        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
        labels = clusterer.fit_predict(X)

        self.df_clusters = pd.DataFrame({'label': labels})

        os.makedirs(self.output_directory, exist_ok=True)

        for cluster_label, cluster_size in self.df_clusters['label'].value_counts().items():
            if cluster_size >= self.min_cluster_size and cluster_label != -1:
                cluster_mask = (labels == cluster_label)

                valid_mask = np.isfinite(data['phot_bp_mean_mag']) & np.isfinite(data['phot_rp_mean_mag'])
                valid_cluster_mask = np.isin(np.arange(len(data)), np.nonzero(cluster_mask)[0]) & valid_mask

                if np.sum(valid_cluster_mask) >= self.min_cluster_size:
                    self.plot_clusters(cluster_label, data, cluster_mask, iso_data)

        print(f'Your results: {self.output_directory}')

    def plot_clusters(self, cluster_label, data, cluster_mask, iso_data):
        fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
        cluster_plotter = ClusterPlotter(ax_2d, data, cluster_mask,
                                        cluster_color=plt.cm.jet(cluster_label / len(self.df_clusters['label'].unique())),
                                        label=f'Cluster {cluster_label}')
        cluster_plotter.plot()

        output_filename_2d = f'{self.output_directory}cluster_{cluster_label}_2d.tif'
        plt.savefig(output_filename_2d, bbox_inches='tight', dpi=300)
        plt.close(fig_2d)

        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        isochrone_plotter = IsochronePlotter(ax_3d, data, cluster_mask,
                                            cluster_color=plt.cm.jet(cluster_label / len(self.df_clusters['label'].unique())),
                                            label=f'Cluster {cluster_label}', iso_data=iso_data)
        isochrone_plotter.plot()

        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=self.df_clusters['label'].min(), vmax=self.df_clusters['label'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_3d, orientation='vertical', fraction=0.03, pad=0.1)
        cbar.set_label('Cluster Label')
        output_filename_3d = f'{self.output_directory}cluster_{cluster_label}_isochrone_3d.tif'
        plt.savefig(output_filename_3d, bbox_inches='tight', dpi=300)
        plt.close(fig_3d)

if __name__ == "__main__":
    fits_file_path = 'Path to stars data'
    iso_file_path = 'Your path to isochrones data'
    output_directory = 'Your path to save results'

    analyzer = ClusterAnalyzer(fits_file_path, iso_file_path, output_directory)
    analyzer.run_analysis()
