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

class TestModelFunc(unittest.TestCase):
    def setUp(self):
        self.test_img_path = os.path.join(cfg.BASE_DIR,'retinopathy_test/tests/input/dr4.tiff')
        self.pred_result = retina_model.predict_file(self.test_img_path)
 
    def test_predict_type(self):
        """
        Test that predict_file() returns dictionary
        """
        self.assertTrue(type(self.pred_result) is dict)
        
    def test_predict_file(self):
        """
        Functional test of predict_file
        Also visualizes the neural network
        """
        test_class = 4
        prob_cut = 0.5

        prob = self.pred_result[self.test_img_path]['probabilities'][0][test_class]
        print(self.pred_result)
        print(self.pred_result[self.test_img_path]['probabilities'])
        print(self.pred_result[self.test_img_path]['classes'])
        #for pred in pred_result["predictions"]:
        #    print("prob: ", pred["probability"])
        #    print("label: ", pred["label"])
        #    label = pred["label"]
        #    if label == 'Saint_bernard':
        #        prob = pred["probability"]
    
        #model = retina_model.build_model(network)

        # print model summary
        #model.summary()
        #print("prob for dr4.tiff: ", prob)
    
        assert prob > prob_cut

if __name__ == '__main__':
    unittest.main()
    #test_predict_file()