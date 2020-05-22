# -*- coding: utf-8 -*-
#from os import path
import os
import tensorflow as tf
from webargs import fields, validate, ValidationError
from marshmallow import Schema, INCLUDE
# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data')
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')

#Retina_RemoteStorage = 'rshare:/deep-oc-apps/retinopathy_test'
Retina_RemoteStorage = 'rshare:/retinopathy_test'
Retina_RemotePublic = 'https://nc.deep-hybrid-datacloud.eu/s/yNsYpACAFbFS8Rp/download?path='
Retina_RemoteDataRecords = os.path.join(Retina_RemoteStorage, 'data', 'records')
Retina_RemoteModelsUpload = os.path.join(Retina_RemoteStorage, 'models')
Retina_LocalDataRecords = os.path.join(DATA_DIR, 'records')
Retina_LocalModels = MODELS_DIR
Retina_LocalModelsServe = os.path.join(Retina_LocalModels, 'retinopathy_serve')
Retina_TrainingData = "retinopathy_tr.tfrecords."    # "retinopathy_tr.tfrecords"
Retina_ValidationData = "retinopathy_va.tfrecords."  # "retinopathy_va.tfrecords"

### it should work on CPU (if num_gpus=0) but very..very slow
#def gpus_must_exist(val):
#    if val > 0:
#        if not tf.test.is_gpu_available():
#            raise ValidationError("GPUs does not exist.")
#
                
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    trained_graph = fields.Str(
        required=False,
        missing='1540408813_cpu',
        enum=['1540408813_cpu', '1533577729_gpu'],
        description="Pre-trained graph to use"
    )

    files = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select the image you want to classify."
    )

    urls = fields.Url(
        required=False,
        missing=None,
        description="Select an URL of the image you want to classify."
    )


# class / place to describe arguments for train()
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # supports extra parameters

    train_epochs = fields.Integer(
        required=False,
        missing=10,
        description="Number of training epochs")

    epochs_between_evals = fields.Integer(
        required=False,
        missing=1,
        description="Number of training epochs between evaluation runs")
    
    batch_size = fields.Integer(
        missing=16,
        description='Global Batch size',
        required=False)

    num_gpus =  fields.Integer(
        missing=1,
        #validate=gpus_must_exist,
        description='Number of GPUs to use, if available (0 = CPU)',
        required=False)
    
    upload_back = fields.Boolean(
        missing=False,
        enum=[False, True],
        description='Either upload a trained graph back to the remote storage (True) or not (False, default)',
        required=False)
