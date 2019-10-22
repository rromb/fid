''' Calculates the Frechet Inception Distance (FID) to evalulate GANs or
other image generating functions.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.
'''

import numpy as np
import os
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
import warnings

from tqdm import tqdm

from edflow.iterators.batches import make_batches
from edflow.data.util import adjust_support
from edflow.util import retrieve
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    

#------------------
# The following methods are implemented to obtain a batched version of the activations.
# This has the advantage to reduce memory requirements, at the cost of slightly reduced efficiency.
# - Pyrestone
#------------------


def get_activations_from_dset(dset, sess, batch_size=50, imkey='image', verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dset        : DatasetMixin which contains the images.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- imkey      : Key at which the images can be found in each example.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = len(dset)
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0

    batches = make_batches(dset, batch_size, shuffle=False)
    n_batches = len(batches)
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))

    print('d0', d0)
    print('n_batches', n_batches)
    print('n_ui', n_used_imgs)

    for i, batch in enumerate(tqdm(batches, desc='FID')):
        if i >= n_batches:
            break
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        images = retrieve(batch, imkey)
        images = adjust_support(np.array(images), '0->255', clip=True)
        images = images.astype(np.float32)[..., :3]

        if images.shape[-1] == 1:
            images = np.tile(images, [1,1,1,3])

        if len(pred_arr[start:end]) == 0:
            continue

        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': images})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
        del batch  # clean up memory

    if verbose:
        print(" done")
    return pred_arr

    
def calculate_activation_statistics_from_dset(dset, sess, batch_size=50, imkey='image', verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- dset        : DatasetMixin which contains the images.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- imkey      : Key at which the images can be found in each example.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations_from_dset(dset, sess, batch_size, imkey, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
    
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def calculate_fid_given_dsets(dsets, imkeys, inception_path,
                              batch_size=50):
    ''' Calculates the FID of two paths. '''
    inception_path = check_or_download_inception(inception_path)

    create_inception_graph(str(inception_path))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        m1, s1 = calculate_activation_statistics_from_dset(
                dsets[0], sess, batch_size, imkeys[0]
                )
        m2, s2 = calculate_activation_statistics_from_dset(
                dsets[1], sess, batch_size, imkeys[1]
                )

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value


def fid(root, data_in, data_out, config,
        im_in_key='image', im_out_key='image', name='fid'):

    incept_p = os.environ.get(
            'INCEPTION_PATH', 
            '/export/scratch/jhaux/Models/inception_fid'
            )
    inception_path = retrieve(config, 'fid/inception_path', default=incept_p)

    batch_size = retrieve(config, 'fid/batch_size', default=50)

    fid_value = calculate_fid_given_dsets(
            [data_in, data_out],
            [im_in_key, im_out_key],
            inception_path,
            batch_size)

    if 'model_output.csv' in root:
        root = root[:-len('model_output.csv')]

    save_dir = os.path.join(root, name)
    savename = os.path.join(save_dir, 'score.txt')

    os.makedirs(save_dir, exist_ok=True)

    with open(savename, 'w+') as f:
        f.write(str(fid_value))

    print('\nFID SCORE: {}'.format(fid_value))

if __name__ == "__main__":
    from edflow.debug import DebugDataset
    from edflow.data.dataset import ProcessedDataset

    D1 = DebugDataset(size=100)
    D2 = DebugDataset(size=100)

    P = lambda *args, **kwargs: {'image': np.ones([256, 256, 3])}

    D1 = ProcessedDataset(D1, P)
    D2 = ProcessedDataset(D2, P)

    print(D1[0])

    fid('.', D1, D2, {})
