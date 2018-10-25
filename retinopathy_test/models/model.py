# -*- coding: utf-8 -*-
"""
Model description
"""
import pkg_resources
# import project config.py
import retinopathy_test.config as cfg
# import run_prediction #ki: comment out to avoid tensorflow import
import os

def get_metadata():

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for l in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if l.startswith(par):
                _, v = l.split(": ", 1)
                meta[par] = v

    return meta

def predict_file(img_path, *args):
    """
    Function to make prediction on a local file
    """
    #print image_path
    #model_dir = os.path.join(cfg.BASE_DIR, 'models','retinopathy_serve')
    #run_prediction.predict_image(model_dir,img_path)
    message = 'Not implemented in the model (predict_file)'
    return message


def predict_data(img_path,*args):
    """
    Function to make prediction on an uploaded file
    """
    print (img_path)
    #model_dir = os.path.join(cfg.BASE_DIR, 'models','retinopathy_serve')
    #run_prediction.predict_image(model_dir,img_path)
    
    message = 'Not implemented in the model (predict_data hello!)'
    return message


def predict_url(*args):
    """
    Function to make prediction on a URL
    """    
    message = 'Not implemented in the model (predict_url)'
    return message


def train(*args):
    """
    Train network
    """
    message = 'Not implemented in the model (train)'
    return message
