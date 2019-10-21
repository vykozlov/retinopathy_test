# -*- coding: utf-8 -*-
from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
Retina_RemoteStorage = 'rshare:/'


train_args = { 'num_epochs': {'default': 1,
                              'help': 'Number of epochs to train on',
                              'required': False
                             },
}
predict_args = {'trained_graph':   {'default': '1540408813',
                             'choices': ['1540408813', '1533577729'],
                             'help': 'Version of the pre-trained graph to use',
                             'required': False
                           },

}
