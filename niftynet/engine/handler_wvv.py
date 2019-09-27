# -*- coding: utf-8 -*-
"""
This module implements a sampler threads controller.
"""
import tensorflow as tf

from niftynet.engine.signal import ITER_STARTED,ITER_FINISHED
from niftynet.utilities.util_common import traverse_nested


class WVV_Monitor(object):
    """
    This class handles iteration events to ensure the images are saved correctly.
    """

    def __init__(self, **_unused):
        # ITER_STARTED.connect(self.check_progess())
        ITER_FINISHED.connect(self.save_images)


    def save_images(self, sender, **msg):
        """
        Stop the sampler's threads

        :param sender: an instance of niftynet.application
        :param _unused_msg:
        :return:
        """
        if sender.action_param.do_whole_volume_validation:
            # tf.logging.info("Number of images processed: {}".format(sender.output_decoder.n_id_changes))
            if sender.output_decoder.num_images_saved == len(sender.readers[1]._file_list):
                tf.logging.info("Processed last image, saving image and switching back to training.")
                sender.output_decoder.n_id_changes = 0
                sender.output_decoder.end_val = True
                sender.output_decoder.num_images_saved = 0
                sender.readers[1].reset()


