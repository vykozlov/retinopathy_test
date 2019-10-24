# -*- coding: utf-8 -*-
"""
Model description
"""
import yaml
import pkg_resources
# import project config.py
import retinopathy_test.config as cfg
import retinopathy_test.models.run_prediction as runpred #ki: comment out to avoid tensorflow import
import os
import tempfile
import retinopathy_test.models.retinopathy_main as retimain
import retinopathy_test.models.resnet_run_loop as resnet_run_loop
# import retinopathy_test.models.models-master.official as official
from absl import flags
from absl import app as absl_app
from official.utils.logs import logger

import tensorflow as tf
#from official.utils.flags import core as flags_core
import deepaas
from pkg_resources import parse_version
import subprocess
import time


def rclone_copy(src_path, dest_path):
    command = (['rclone', 'copy', '--progress', src_path, dest_path])    
    try:
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
    except OSError as e:
        output, error = None, e
    return output, error
    
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

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par+":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta

def predict_file(img_path, trained_graph):
    """
    Function to make prediction on a local file
    """
    print ('[DEBUG] image_path: ', img_path)
    model_dir = os.path.join(cfg.Retina_LocalModelsServe, trained_graph)
    print ('[DEBUG] model_dir: ', model_dir)

    if not os.path.exists(model_dir):
        output, error = rclone_copy(os.path.join(cfg.Retina_RemoteModelsServe, trained_graph),
                                    model_dir)

    results=runpred.predict_image(model_dir, img_path)
    print ('[DEBUG] results: %s'%results)
    return results


def predict_data(*args, **kwargs):
    """
    Function to make prediction on an uploaded file
    """
    deepaas_ver_cut = parse_version('0.5.0')
    imgs = []
    filenames = []
    
    deepaas_ver = parse_version(deepaas.__version__)
    print("[INFO] deepaas_version: %s" % deepaas_ver)
    predict_debug = False
    if predict_debug:
        print('[DEBUG] predict_data - args: %s' % args)
        print('[DEBUG] predict_data - kwargs: %s' % kwargs)
    if deepaas_ver >= deepaas_ver_cut:
        for arg in args:
            imgs.append(arg['files'])
            trained_graph = str(yaml.safe_load(arg.trained_graph))
    else:
        imgs = args[0]

    if not isinstance(imgs, list):
        imgs = [imgs]
            
    for image in imgs:
        if deepaas_ver >= deepaas_ver_cut:
            f = open("/tmp/%s" % image[0].filename, "w+")
            image[0].save(f.name)
        else:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(image)
        f.close()
        filenames.append(f.name)
        print("Stored tmp file at: {} \t Size: {}".format(f.name,
        os.path.getsize(f.name)))

    prediction = []
    try:
        for imgfile in filenames:
            prediction.append(str(predict_file(imgfile, trained_graph)))
            print("image: ", imgfile)
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


def train(train_args):
    """
    Train network
    """
    # Check if necessary local directories exist:
    if not os.path.exists(cfg.Retina_LocalDataRecords):
        print("[INFO] %s is not found locally, creating..." % 
              cfg.Retina_LocalDataRecords)
        os.makedirs(cfg.Retina_LocalDataRecords)
    if not os.path.exists(cfg.Retina_LocalModelsServe):
        print("[INFO] %s is not found locally, creating..." % 
              cfg.Retina_LocalModelsServe)
        os.makedirs(cfg.Retina_LocalModelsServe)  

    # Take parameters defined via deepaas by a user
    #num_gpus = yaml.safe_load(train_args.num_gpus)
    train_epochs = yaml.safe_load(train_args.train_epochs)
    batch_size = yaml.safe_load(train_args.batch_size)
    
    # from deep-nextcloud into the container
    # data_origin = 'rshare:/records_short/'
    training_data = os.path.join(cfg.Retina_LocalDataRecords, 
                                 cfg.Retina_TrainingData)
    validation_data = os.path.join(cfg.Retina_LocalDataRecords, 
                                   cfg.Retina_ValidationData)
    e1=time.time()
    # check if retinopathy_tr.tfrecord or retinopathy_va.tfrecord exist locally,
    # if not -> download them from the RemoteStorage
    if not (os.path.exists(training_data) or os.path.exists(validation_data)):
        # Retina_RemoteDataRecords and Retina_LocalDataRecords are defined in config.py #vk
        print("[INFO] Either %s or %s NOT found locally, download them from %s" % 
              (training_data, validation_data, cfg.Retina_RemoteDataRecords))
        output, error = rclone_copy(cfg.Retina_RemoteDataRecords, cfg.Retina_LocalDataRecords)
        print(error)
        
    download_time=time.time()-e1
    time.sleep(60)
 
    FLAGS=flags.FLAGS
    #flags.DEFINE_string('listen-ip', '0.0.0.0', 'port')
    #flags.FLAGS.unparse_flags()
    for name in list(FLAGS):
        print (name)
        #if name == 'listen-ip':
            #delattr(flags.FLAGS, name)

    e2=time.time()
    tf.logging.set_verbosity(tf.logging.INFO)
    print("[DEBUG] Logging verbosity set")
    #flags.adopt_module_key_flags(retimain)
    print("[DEBUG] Flags adopted from resnet_run_loop")
    #absl_app.run(retimain.define_retinopathy_flags(
    #                                        batch_size=batch_size,
    #                                        train_epochs=train_epochs)) #FLAGS
    #FLAGS=flags.FLAGS
    #for name in list(FLAGS):
    #    print (name)

    print("[DEBUG] Flags define_retinopaty set")
    #absl_app.run(retimain.main())

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
    code = subprocess.call(["python", training_script, 
                            '--batch_size', str(batch_size), 
                            '--train_epochs', str(train_epochs)])
    print(code)
    training_time=time.time()-e2
    time.sleep(60)

    e3=time.time()
    # Retina_LocalModelsServe and Retina_RemoteModelServe are defined in config.py #vk
    upload_back = yaml.safe_load(train_args.upload_back)
    if(upload_back):
        def getmtime(name):
            path = os.path.join(cfg.Retina_LocalModelsServe, name)
            return os.path.getmtime(path)
        # build list of directories aka training runs
        train_runs = sorted(os.listdir(cfg.Retina_LocalModelsServe),
                            key=getmtime, reverse=True)
        print("[DEBUG] train_runs: ", train_runs)
        last_run = train_runs[0]
        print("[DEBUG] last run: ", last_run)
        # copy only last training run
        output, error = rclone_copy(os.path.join(cfg.Retina_LocalModelsServe, last_run),
                                    os.path.join(cfg.Retina_RemoteModelsServe, last_run))
        print(error)

    upload_time=time.time()-e3
    training_data_path = os.path.join(cfg.Retina_LocalDataRecords,
                                      'retinopathy_tr.tfrecords')
    #message = 'Not implemented in the model (train)'
    message = 'Training finished! download time: %f, training time: %f, upload time: %f, training file size: %s'%(download_time,training_time,upload_time,file_size(training_data_path))
    return message

def get_train_args():

    train_args = cfg.train_args

    # convert default values and possible 'choices' into strings
    for key, val in train_args.items():
        val['default'] = str(val['default']) #yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]

    return train_args


# !!! deepaas>=0.5.0 calls get_test_args() to get args for 'predict'
def get_test_args():
    predict_args = cfg.predict_args

    # convert default values and possible 'choices' into strings
    for key, val in predict_args.items():
        val['default'] = str(val['default'])  # yaml.safe_dump(val['default']) #json.dumps(val['default'])
        if 'choices' in val:
            val['choices'] = [str(item) for item in val['choices']]
        #print(val['default'], type(val['default']))

    return predict_args
