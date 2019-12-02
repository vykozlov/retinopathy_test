import os, argparse, sys

import tensorflow as tf
import numpy as np
import glob

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))

import cv2
def load_image(addr, img_size):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    #record pipeline
    img = cv2.imread(addr)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #COLOR_BGR2RGB
    img = img.astype(np.float32)

    #iterator pipeline
    img = img / 255.0
    img = img - 0.5
    return img.reshape([1,256,256,3])

def predict_image(model_dir, image_file): #ki: this is the main function performing prediction
    # f = glob.glob(model_dir + '*/')[0] #ki: get the model parameters from the first subdirectory in model_dir
    # f = glob.glob(model_dir + '1540408813/')[0] #ki: directory 1540408813 contains model parameters for cpu only machine
    # @ki: if you like to use different trained models, we can pass them as a parameter to the function
    #      i.e. skip adding '154048813' as a fixed parameter here #vk
    f = glob.glob(model_dir)[0]
    print (f)
    imgs = image_file.split(',')
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir = f, signature_def_key='predict')#ki: create predictor function using the graph and model parameters
    results={}
    for imgfile in imgs:
        img = load_image(imgfile, 256)
        output = predictor_fn({'input': img})
        print(imgfile, output)
        results['%s'%imgfile]=output
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #ki: create command line parser to get model parameters and image files.
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./retinopathy_serve/',
        # default='./',
        help="""\
        Path to classify_image_graph_def.pb\
        """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='dr4.tiff',
        help='Absolute path to image file.'
    )    
    args = parser.parse_args()
    results = predict_image(args.model_dir, args.image_file)#ki: take the model parameter and input images and return predicted class (probabilities)
