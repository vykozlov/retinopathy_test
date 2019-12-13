# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat Aug 10 08:47:51 2019

@author: vykozlov
"""
import unittest
import numpy as np
import retinopathy_test.models.model as retina_model

#from keras import backend as K

debug = True

class TestModelMethods(unittest.TestCase):
    def setUp(self):
        self.meta = retina_model.get_metadata()
        
    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns list
        """
        self.assertTrue(type(self.meta) is dict)
        
    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns 
        right values (subset)
        """
        self.assertEqual(self.meta['Name'].replace('-','').replace('_',''),
                         'retinopathy_test'.replace('-','').replace('_',''))
        self.assertEqual(self.meta['Author'], 'HMGU')
        self.assertEqual(self.meta['Author-email'], 'itokeiic@gmail.com')


if __name__ == '__main__':
    unittest.main()
