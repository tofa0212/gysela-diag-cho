import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def list_datasets_and_groups(h5file, indent=0):
    """Recursively list datasets and groups in an HDF5 file."""
    for key in h5file.keys():
        item = h5file[key]
        if isinstance(item, h5py.Dataset):  # Check if the current item is a dataset
            if item.shape == () or item.shape == (1,):  # Check if it's a scalar dataset
                print("  " * indent + key + "  " + str(np.squeeze(item)))  # Use empty tuple to read scalar value
            else:
                print("  " * indent + key + "  " + str(item.shape))

        elif isinstance(item, h5py.Group):  # Check if the current item is a group
            print("  " * indent + key)  # Just print the group name
            list_datasets_and_groups(item, indent + 1)


def find_all_datasets(h5file, name, datasets):
    for key in h5file.keys():
        item = h5file[key]
        if isinstance(item, h5py.Dataset):
            if key == name:
                datasets.append(np.squeeze(item))
        elif isinstance(item, h5py.Group):
            find_all_datasets(item, name, datasets)

def plot_data(datasets, dataset_names):
    num_datasets = len(datasets)
    if all(dat.ndim == 1 for dat in datasets):
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size':14})
        plt.set_cmap('tab10')
        for dat, name in zip(datasets, dataset_names):
            plt.plot(dat[:], label=name['dn'])
        plt.legend()

        plt.tight_layout()
        plt.show()
    elif all(dat.ndim > 1 for dat in datasets):
        fig, axs = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 5))
        if num_datasets == 1:
            axs = [axs]
        for i, (dat, name) in enumerate(zip(datasets, dataset_names)):
            cax = axs[i].contourf(dat, levels=50, cmap='viridis')
            print(np.where(dat <0))
            axs[i].set_title(name)
            fig.colorbar(cax, ax=axs[i])
        plt.show()

def main(filename, dataset_names):
    datasets = []
    with h5py.File(filename, 'r') as f:
        list_datasets_and_groups(f)
        for name in dataset_names:
            find_all_datasets(f, name['d'], datasets)

    if not dataset_names:  #
        print("No datasets specified for plotting.")
        return                

    plot_data(datasets, dataset_names)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot data from HDF5 file.")
    parser.add_argument('filename', type=str, help='Name of the HDF5 file.')

    parser.add_argument('-d', action='append', dest='dopt', \
                        type=lambda kv: kv.split(":"), help='Dataset name followed by optional display name after ":".', default=[])

    args = parser.parse_args()
    dopt = []
    for opt in args.dopt:
        opt_dict = {'d':opt[0]}
        if len(opt) > 1:
            opt_dict['dn'] = opt[1]
        else:
            opt_dict['dn']= opt[0]
        dopt.append(opt_dict)


    main(args.filename, dopt)
