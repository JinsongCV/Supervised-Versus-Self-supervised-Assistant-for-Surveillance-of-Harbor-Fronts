# README #

This repo lets you train a basic autoencoder and use it for anormaly detection.

### Overview ###
* define_crop_region.py place the corner boxes of the ROI
* extract_datasets.py extracts crops from a specific view point.
* train.py trains and tests a model using the extracted crops.
* embed.py lets load a model and output embeddings of the extracted crops.
* plot_distribution.py

### prepare data ###

There are two options. (1) using prepared data(crops) as done for the "Supervised Versus Self-supervised Assistant for Surveillance of Harbor Fronts" paper or (2) extract specific views and crops using "extract_datasets.py".

To use pre-cropped data simply place the image files in data/[test,train].

In you want to use "extract_datasets.py":

* Choose view point "view" variable and determine whether the other options are correct.
* Set flow=False if you want just the [intensity] otherwise crops are stored containing [intensity, flow_x, flow_y].

### train ###

* train mode, no surprises.

### embed ###

* produces reconstructions, reconstruction losses, and latent representation and stores it as .npy.

### plot_distribution ###

* Loads numpy arrays from the .npy file, fits PCA to latent representation vectors, and visualizes the distribution in 2D using the two most significant components.
