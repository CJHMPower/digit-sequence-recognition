import os, struct
from array import array as pyarray 
from numpy import append, array, int8, uint8, zeros
import gzip
import numpy as np
import tarfile
import numpy as np
import h5py
from PIL import Image

def unzip_and_write(input_file, output_file, path):
	with gzip.open(path+input_file, 'rb') as f:
		outF = open(path+output_file, 'wb')
		outF.write( f.read())
		f.close()
		outF.close()
def untar_and_write(input_file, path):
	tar = tarfile.open(input_file, "r:gz")
	tar.extractall(path = path)
	tar.close()
def unzip_mnist(dataset = "training", path="./"):
	if dataset == "testing":
		unzip_and_write('t10k-labels-idx1-ubyte.gz','t10k-labels-idx1-ubyte', path)
		unzip_and_write('t10k-images-idx3-ubyte.gz','t10k-images-idx3-ubyte', path)
	elif dataset == "training":
		unzip_and_write('train-images-idx3-ubyte.gz','train-images-idx3-ubyte', path)
		unzip_and_write('train-labels-idx1-ubyte.gz','train-labels-idx1-ubyte', path)
def untar_svhn(dataset = "training",  path="./"):
	if dataset == "testing":
		untar_and_write(path + 'test.tar.gz', path[:-1])
	elif dataset == "training":
		untar_and_write(path + 'train.tar.gz', path[:-1])
def load_mnist(dataset="training", digits=None, path=None, asbytes=False, selection=None, return_labels=True, return_indices=False):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. It is recommended
    to set the environment variable ``MNIST`` to point to the folder where you
    put the data, so that you don't have to select path. On a Linux+bash setup,
    this is done by adding the following to your ``.bashrc``::

        export MNIST=/path/to/mnist

    Parameters
    ----------
    dataset : str 
        Either "training" or "testing", depending on which dataset you want to
        load. 
    digits : list 
        Integer list of digits to load. The entire database is loaded if set to
        ``None``. Default is ``None``.
    path : str 
        Path to your MNIST datafiles. The default is ``None``, which will try
        to take the path from your environment variable ``MNIST``. The data can
        be downloaded from http://yann.lecun.com/exdb/mnist/.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to
        ``numpy.float64`` in [0.0, 1.0].
    selection : slice
        Using a `slice` object, specify what subset of the dataset to load. An
        example is ``slice(0, 20, 2)``, which would load every other digit
        until--but not including--the twentieth.
    return_labels : bool
        Specify whether or not labels should be returned. This is also a speed
        performance if digits are not specified, since then the labels file
        does not need to be read at all.
    return_indicies : bool
        Specify whether or not to return the MNIST indices that were fetched.
        This is valuable only if digits is specified, because in that case it
        can be valuable to know how far
        in the database it reached.

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. If neither labels nor inices are returned, then this is returned directly, and not inside a 1-sized tuple.
    labels : ndarray
        Array of size ``N`` describing the labels. Returned only if ``return_labels`` is `True`, which is default.
    indices : ndarray
        The indices in the database that were returned.

    Examples
    --------
    Assuming that you have downloaded the MNIST database and set the
    environment variable ``$MNIST`` point to the folder, this will load all
    images and labels from the training set:

    >>> images, labels = ag.io.load_mnist('training') # doctest: +SKIP

    Load 100 sevens from the testing set:    

    >>> sevens = ag.io.load_mnist('testing', digits=[7], selection=slice(0, 100), return_labels=False) # doctest: +SKIP

    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    if path is None:
        try:
            path = os.environ['MNIST']
        except KeyError:
            raise ValueError("Unspecified path requires environment variable $MNIST to be set")

    try:
        images_fname = os.path.join(path, files[dataset][0])
        labels_fname = os.path.join(path, files[dataset][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    # We can skip the labels file only if digits aren't specified and labels aren't asked for
    if return_labels or digits is not None:
        flbl = open(labels_fname, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())
        flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in range(size) if labels_raw[k] in digits]
    else:
        indices = range(size)

    if selection:
        indices = indices[selection] 
    N = len(indices)

    images = zeros((N, rows, cols), dtype=uint8)

    if return_labels:
        labels = zeros((N), dtype=int8)
    for i, index in enumerate(indices):
        images[i] = array(images_raw[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ]).reshape((rows, cols))
        if return_labels:
            labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(float)/255.0

    ret = (images,)
    if return_labels:
        ret += (labels,)
    if return_indices:
        ret += (indices,)
    if len(ret) == 1:
        return ret[0] # Don't return a tuple of one
    else:
        return ret

def one_hot_encode_labels(labels):
    values = labels.astype(np.int16)
    n_values = np.max(values) + 1
    labels_encoded = np.eye(n_values)[values]
    return labels_encoded

# from http://codegists.com/snippet/python/read_svhn_matpy_veeresht_python (edited)
def read_process_h5(filename):
    """ Reads and processes the mat files provided in the SVHN dataset. 
        Input: filename 
        Ouptut: list of python dictionaries 
    """
         
    f = h5py.File(filename, 'r')
    groups = f['digitStruct'].items()
    bbox_ds = np.array(groups[0][1]).squeeze()
    names_ds = np.array(groups[1][1]).squeeze()
 
    data_list = []
    num_files = bbox_ds.shape[0]
    count = 0
 
    for objref1, objref2 in zip(bbox_ds, names_ds):
 
        data_dict = {}
 
        # Extract image name
        names_ds = np.array(f[objref2]).squeeze()
        filename = ''.join(chr(x) for x in names_ds)
        data_dict['filename'] = filename
 
        #print filename
 
        # Extract other properties
        items1 = f[objref1].items()
 
        # Extract image label
        labels_ds = np.array(items1[1][1]).squeeze()
        try:
            label_vals = [int(f[ref][:][0, 0]) for ref in labels_ds]
        except TypeError:
            label_vals = [labels_ds]
        data_dict['labels'] = label_vals
        data_dict['length'] = len(label_vals)
 
        # Extract image height
        height_ds = np.array(items1[0][1]).squeeze()
        try:
            height_vals = [f[ref][:][0, 0] for ref in height_ds]
        except TypeError:
            height_vals = [height_ds]
        data_dict['height'] = height_vals
 
        # Extract image left coords
        left_ds = np.array(items1[2][1]).squeeze()
        try:
            left_vals = [f[ref][:][0, 0] for ref in left_ds]
        except TypeError:
            left_vals = [left_ds]
        data_dict['left'] = left_vals
 
        # Extract image top coords
        top_ds = np.array(items1[3][1]).squeeze()
        try:
            top_vals = [f[ref][:][0, 0] for ref in top_ds]
        except TypeError:
            top_vals = [top_ds]
        data_dict['top'] = top_vals
 
        # Extract image width
        width_ds = np.array(items1[4][1]).squeeze()
        try:
            width_vals = [f[ref][:][0, 0] for ref in width_ds]
        except TypeError:
            width_vals = [width_ds]
        data_dict['width'] = width_vals
 
        data_list.append(data_dict)
 
        count += 1
 
    return data_list

def get_bounding_box(image_dict):
    top = min(image_dict['top'])
    left = min(image_dict['left'])
    
    top_and_height = [image_dict['top'], image_dict['height']]
    left_and_width = [image_dict['left'], image_dict['width']]
    
    bottom =max(np.sum(top_and_height, axis=0))
    right = max(np.sum(left_and_width,axis=0))
    
    return {'top': top, 'bottom': bottom, 'left':left, 'right':right}

def crop_and_resize(image_path, dimensions_dict):
    original = Image.open(image_path)
    left = int(dimensions_dict['left'])
    top = int(dimensions_dict['top'])
    right = int(dimensions_dict['right'])
    bottom =int(dimensions_dict['bottom'])
    box = (left,top,right,bottom)
    cropped = original.crop(box)
    resized = cropped.resize((54, 54))
    return resized

def process_all_images(source_folder, destination_folder, images_info):
    images_list = os.listdir(source_folder)
    images_list.remove('digitStruct.mat')
    images_list.remove('see_bboxes.m')
    for image_name in images_list:
        image_num = int(image_name[:-4]) - 1
        processed = crop_and_resize(source_folder+image_name, get_bounding_box(images_info[image_num]))
        processed.save(destination_folder + image_name) 
