import DLStudio
import os
import unittest

class TestInstanceCreation(unittest.TestCase):

    def setUp(self):
        self.dls = DLStudio.DLStudio(
                                     convo_layers_config = "1x[128,7,7,1]-MaxPool(2)",
                                     fc_layers_config = [-1,1024,10],
                                     image_size = [32,32]
                                    )
 

    def test_instance_creation(self):
        print("testing instance creation")
        convo_configs  =    self.dls.parse_config_string_for_convo_layers()
        self.assertEqual(len(convo_configs), 1)

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestInstanceCreation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

