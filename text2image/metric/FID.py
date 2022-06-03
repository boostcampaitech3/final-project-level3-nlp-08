# pip install keras tensorflow scipy NEEDED
# calculating the frechet inception distance between two images (NOT datasets)
import argparse
from PIL import Image
import numpy as np
import scipy as sp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
 
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)

def compute_embeddings(model, images):
    image_embeddings = []
    if len(images) == 1: # run if single image input
        embeddings = model.predict(images)
        for _ in range(10000): # exaggerating data
            image_embeddings.extend(embeddings)
    else: # run if several images input
        for image in images:
            embeddings = model.predict(image)
            image_embeddings.extend(embeddings)

    return np.array(image_embeddings)

# calculate fid
def calculate_fid(image1, image2):
    # calculate mean and covariance statistics
    mu1, sigma1 = image1.mean(axis=0), np.cov(image1, rowvar=False)
    mu2, sigma2 = image2.mean(axis=0), np.cov(image2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sp.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

def main(args):
    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    # load images (additional feature is needed in this part if you want to put lots of data)
    img1, img2  = Image.open(args.origin_path).convert('RGB'), Image.open(args.target_path).convert('RGB')
    image1, image2 = np.asarray(img1), np.asarray(img2)
    image1, image2 = np.array([image1]), np.array([image2])
    print('Loaded', image1.shape, image2.shape)
    # convert integer to floating point values
    image1 = image1.astype('float32')
    image2 = image2.astype('float32')
    # resize images
    image1 = scale_images(image1, (299,299,3))
    image2 = scale_images(image2, (299,299,3))
    print('Scaled', image1.shape, image2.shape)
    # pre-process images
    image1 = preprocess_input(image1)
    image2 = preprocess_input(image2)

    image1 = compute_embeddings(model, image1)
    image2 = compute_embeddings(model, image2)
    
    fid = calculate_fid(image1, image2)
    print('FID: %.3f' % fid)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", dest="origin_path", default="../assets/answer_image.png")
    parser.add_argument("--path2", dest="target_path", default="../assets/generated_image.png")
    args = parser.parse_args()
    main(args)