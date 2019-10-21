# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat Aug 10 08:43:00 2019

@author: vykozlov
"""
import os
import unittest
import retinopathy_test.config as cfg
import retinopathy_test.models.model as retina_model

import flask
import flask_restplus


import deepaas
#from deepaas import api
from deepaas.api import v1
import deepaas.model
#from deepaas.tests import base


class TestModelFunc(unittest.TestCase):
    def setUp(self):
        self.test_img = "dr4.tiff"
        self.test_img_dir = os.path.join(cfg.BASE_DIR,'retinopathy_test/tests/input/')
        self.test_img_path = os.path.join(self.test_img_dir, self.test_img)
        self.pred_result = retina_model.predict_file(self.test_img_path, "1540408813_cpu")
       
        app = flask.Flask(__name__)
        app.config['TESTING'] = True
        app.config['DEBUG'] = True

        api = flask_restplus.Api(app, doc=False)
        api.add_namespace(v1.api)

        deepaas.model.register_models()

        self.app = app.test_client()
        self.assertEqual(app.debug, True)
 
    def test_predict_type(self):
        """
        Test that predict_file() returns dictionary
        """
        self.assertTrue(type(self.pred_result) is dict)
        
    def test_predict_file(self):
        """
        Functional test of predict_file
        """
        test_class = 4
        prob_cut = 0.5

        prob = self.pred_result[self.test_img_path]['probabilities'][0][test_class]
    
        assert prob > prob_cut
        
    def test_predict_data(self):
        """
        Functional test of predict_data
        Post the input file via flask
        """
        f = open(self.test_img_path, 'rb')
        ret = self.app.post(
            "/models/retinopathy_test/predict",
            data={"data": (f, self.test_img)})
        f.close()
        self.assertEqual(200, ret.status_code)

if __name__ == '__main__':
    unittest.main()
    #test_predict_file()