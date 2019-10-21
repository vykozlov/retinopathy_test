# -*- coding: utf-8 -*-
from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
Retina_RemoteStorage = 'rshare:/'
Retina_RemoteShare = 'https://nc.deep-hybrid-datacloud.eu/s/D7DLWcDsRoQmRMN/download?path=%2F&files='

train_args = { 'num_epochs': {'default': 1,
                              'help': 'Number of epochs to train on',
                              'required': False
                             },
}
predict_args = {'trained_graph':   {'default': '1540408813_cpu',
                             'choices': ['1540408813_cpu', '1533577729_gpu'],
                             'help': 'Pre-trained graph to use',
                             'required': False
                           },

}
