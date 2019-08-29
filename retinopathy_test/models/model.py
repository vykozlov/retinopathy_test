# -*- coding: utf-8 -*-
"""
Model description
"""
import pkg_resources
# import project config.py
import retinopathy_test.config as cfg
import retinopathy_test.models.run_prediction as runpred #ki: comment out to avoid tensorflow import
import os
import tempfile
import retinopathy_test.models.retinopathy_main as retimain
# import retinopathy_test.models.models-master.official as official
from absl import flags
from absl import app as absl_app

import tensorflow as tf
#from official.utils.flags import core as flags_core
import deepaas
import subprocess
import time

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
    #print (img_path)
    #model_dir = os.path.join(cfg.BASE_DIR, 'models','retinopathy_serve')
    model_dir = os.path.join(cfg.BASE_DIR,
                              'retinopathy_test',
                              'models','retinopathy_serve')
    print (model_dir)
    results=runpred.predict_image(model_dir,img_path)
    
    message = 'Not implemented in the model (predict_file, yoohoo!)'
    return results


def predict_data(*args, **kwargs):
    """
    Function to make prediction on an uploaded file
    """
    deepaas_vers_cut = '0.4.0'
    img = []
    filenames = []
    
    deepaas_ver = deepaas.__version__
    print("[INFO] deepaas_version: %s" % deepaas_ver)
    if parse_version(deepaas_ver) > parse_version(deepaas_ver_cut):
        for arg in args:
            filenames.append(arg.files)
            # network = yaml.safe_load(arg.network)
    else:
        # Implementation used with DEEPaaS API <=0.4.0
        img = args[0]
    
        if not isinstance(img, list):
            img = [img]
        # print (img) 
        filenames = []
            
        for image in img:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(image)
            f.close()
            filenames.append(f.name)
            print("tmp file: ", f.name)

    prediction = []
    try:
        for imgfile in filenames:
            prediction.append(str(predict_file(imgfile)))
    except Exception as e:
        raise e
    finally:
        for imgfile in filenames:
            os.remove(imgfile)

    return prediction

    #print (img_path)
    #model_dir = os.path.join(cfg.BASE_DIR, 'models','retinopathy_serve')
    #model_dir = os.path.join('.','retinopathy_serve')
    #model_dir+='/'
    
    #print(model_dir)
    #run_prediction.predict_image(model_dir,img_path)
    
    #message = 'Not implemented in the model (predict_data hello!)'
    #return runpred.predict_image(model_dir,img_path)


def predict_url(*args):
    """
    Function to make prediction on a URL
    """    
    message = 'Not implemented in the model (predict_url)'
    return message

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


def train(*args):
    """
    Train network
    """
    # from deep-nextcloud into the container
    # data_origin = 'rshare:/records_short/'
    e1=time.time()
    data_origin = 'rshare:/records/'
    data_copy = os.path.join(cfg.BASE_DIR,
                              'retinopathy_test',
                              'dataset','records')
    command = (['rclone', 'copy', data_origin, data_copy])    
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    print(error)
    download_time=time.time()-e1
    time.sleep(60)
 
    #FLAGS=flags.FLAGS
    #flags.DEFINE_string('listen-ip', '0.0.0.0', 'port')
    #flags.FLAGS.unparse_flags()
    #for name in list(FLAGS):
        #print (name)
        #if name == 'listen-ip':
            #delattr(flags.FLAGS, name)
    tf.logging.set_verbosity(tf.logging.INFO)
    e2=time.time()
    #retimain.define_retinopathy_flags()#FLAGS
    #absl_app.run(retimain.main)
    #retimain.main(flags)
    #training_script=os.path.join(cfg.BASE_DIR,
                              #'retinopathy_test',
                              #'models',
                              #'retinopathy_main_short.py')
    training_script=os.path.join(cfg.BASE_DIR,
                              'retinopathy_test',
                              'models',
                              'retinopathy_main.py')

    print(training_script)
    code = subprocess.call(["python", training_script])
    print(code)
    training_time=time.time()-e2
    time.sleep(60)
    e3=time.time()
    #data_origin = os.path.join(cfg.BASE_DIR,
                              #'retinopathy_test',
                              #'models',
                              #'retinopathy_serve_short')
    #data_copy = 'rshare:/retinopathy_serve_short/'
    data_origin = os.path.join(cfg.BASE_DIR,
                              'retinopathy_test',
                              'models',
                              'retinopathy_serve')
    data_copy = 'rshare:/retinopathy_serve/'
    
    command = (['rclone', 'copy', data_origin, data_copy])
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    print(error)
    upload_time=time.time()-e3
    training_data_path = os.path.join(cfg.BASE_DIR,
                              'retinopathy_test',
                              'dataset','records','retinopathy_tr.tfrecords')
    #message = 'Not implemented in the model (train)'
    message = 'Training finished! download time: %f, training time: %f, upload time: %f, training file size: %s'%(download_time,training_time,upload_time,file_size(training_data_path))
    return message

def get_train_args():
    #{ 'arg1' : {'default': '1',     #value must be a string (use json.dumps to convert Python objects)
                #'help': '',         #can be an empty string
                #'required': False   #bool
                #},
    #'arg2' : {...
                #},
    #...
    #}
    args = {}
    
    return args

def get_test_args():
    test_args={}
    return test_args
