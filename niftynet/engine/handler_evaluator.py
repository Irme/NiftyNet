# -*- coding: utf-8 -*-
"""
This module implements a sampler threads controller.
"""
import tensorflow as tf
import os
import nibabel as nib
import numpy as np

from niftynet.engine.signal import ITER_STARTED, ITER_FINISHED, SESS_STARTED, \
    SESS_FINISHED
from niftynet.evaluation.pairwise_measures import PairwiseMeasures
from niftynet.layer.pad import PadLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer


class EvaluationHandler(object):
    """
    This class handles iteration events to start/stop samplers' threads.
    """

    def __init__(self, **_unused):
        self.internal_validation_flag = None
        ITER_FINISHED.connect(self.check_end_of_val_and_evaluate)

    def check_end_of_val_and_evaluate(self, _sender, **_msg):
        """
        This function will check the dice score between the whole volume preds
        and the labels for segmentation applications
        :param _sender: application
        :param _msg: iteration message
        :return:
        """
        # pass
        iter_msg = _msg['iter_msg']

        if _sender.is_whole_volume_validating and iter_msg.is_validation:
            self.internal_validation_flag = True
        else:
            if self.internal_validation_flag:
                # In the last iteration we were in validation.
                # Now we should evaluate
                save_seg_dir = _sender.action_param.save_seg_dir
                for i in range(0, len(_sender.readers[-1].output_list)):
                    label = _sender.readers[-1](idx=i)[1]['label']
                    for layer in reversed(_sender.readers[-1].preprocessors):
                        if isinstance(layer, PadLayer):
                            label, _ = layer.inverse_op(label)
                        if isinstance(layer, DiscreteLabelNormalisationLayer):
                            label, _ = layer.inverse_op(label)
                    output = nib.load(
                        os.path.join(save_seg_dir,
                                     'window_seg_' +
                                     _sender.readers[-1]
                                     .get_subject_id(i) + '__wvv_out.nii.gz')) \
                        .get_data()
                    try:
                        dice = \
                            PairwiseMeasures(
                                seg_img=np.where(output > 0, 1, 0),
                                ref_img=np.where(label > 0, 1, 0)).dice_score()
                        tf.logging.info(
                            'subject_id: {} dice: {}'.format(
                                _sender.readers[-1].get_subject_id(i), dice))
                    except ValueError:
                        print('Shapes of prediction and label are'
                              ' mismatched: {} {}'.format(output.shape,
                                                          label.shape))
                self.internal_validation_flag = False
                # Match seg to ground truth
                # Compute PairwiseMeasures
                # Save results to TensorBoard/CSV
