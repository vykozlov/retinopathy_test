# -*- coding: utf-8 -*-
from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
Retina_RemoteStorage = 'rshare:/'
Retina_RemoteShare = 'https://nc.deep-hybrid-datacloud.eu/s/D7DLWcDsRoQmRMN/download?path=%2F&files='
Retina_RemoteDataRecords = path.join(Retina_RemoteStorage, 'records')
Retina_RemoteModelServe = path.join(Retina_RemoteStorage, 'retinopathy_serve_vk')
Retina_LocalDataRecords = path.join(BASE_DIR, 'data', 'records')
Retina_LocalModels = path.join(BASE_DIR, 'models')
Retina_LocalModelServe = path.join(Retina_LocalModels, 'retinopathy_serve')
Retina_TrainingData = "retinopathy_tr.tfrecords"
Retina_ValidationData = "retinopathy_va.tfrecords"

train_args = { 'train_epochs': {'default': 10,
                              'help': 'Number of epochs to train on',
                              'required': False
                             },
               'batch_size': {'default': 16,
                              'help': 'Global Batch size',
                              'required': False
                              },
               'upload_back': {'default': False,
                               'choices': [False, True],
                               'help': 'Either upload a trained graph back to the remote storage (True) or not (False, default)',
                               'required': False
                              },
}
predict_args = {'trained_graph':   {'default': '1540408813_cpu',
                             'choices': ['1540408813_cpu', '1533577729_gpu'],
                             'help': 'Pre-trained graph to use',
                             'required': False
                           },

}
