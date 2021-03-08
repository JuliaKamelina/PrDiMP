import os
import cv2
import math
import numpy as np
import time
import tensorflow as tf
import torch
import torch.nn.functional as F

from .feature_extraction import PrDiMPFeatures, augmentation
from .localization import localize_target, refine_target_box
from .prdimp_tracker import PrDiMPTracker
from .runfiles import settings
from .utils import TensorList, plot_graph, process_regions
from .meta_updater import tclstm, tcopts
from .metric_model import me_extract_regions, ft_net


class PrDiMPMUTracker(PrDiMPTracker):
    def __init__(self, image_sz, net_path, is_color=True, mu_model_dir=None, 
                 mu_model_checkpoint=None, metric_model_path='metric_model/model/metric_model.pt'):
        super().__init__(image_sz, net_path, is_color)
        self.mu_model_checkpoint = mu_model_checkpoint
        self.i = 0
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=tfconfig)
        self.mu_model_dir = mu_model_dir
        self.metric_model_path = metric_model_path

        # init_gt = [*(self.pos[[1,0]] - (self.target_sz[[1,0]] - 1)/2), *self.target_sz[[1,0]]]
        # self.last_gt = [init_gt[1], init_gt[0], init_gt[1] + init_gt[3], init_gt[0] + init_gt[2]]  # ymin xmin ymax xmax

    def initializing(self, image, seq, reuse=False):
        out = self.initialize(image, seq)
        init_gt = [*(self.pos[[1,0]] - (self.target_sz[[1,0]] - 1)/2), *self.target_sz[[1,0]]]
        self.last_gt = [init_gt[1], init_gt[0], init_gt[1] + init_gt[3], init_gt[0] + init_gt[2]]  # ymin xmin ymax xmax
        self.tc_init(self.mu_model_dir, reuse)
        self.metric_init(image, np.array(init_gt), self.metric_model_path)
        self.dis_record = []
        self.state_record = []
        self.rv_record = []
        self.all_map = []

        local_state1, self.score_map, update, self.score_max, dis = self.local_track(image)
        self.pos = torch.FloatTensor(
            [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
        self.target_sz = torch.FloatTensor(
            [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])
        
        return out

    def tc_init(self, model_dir, reuse=False):
        self.tc_model = tclstm()
        self.X_input = tf.placeholder("float", [None, tcopts['time_steps'], tcopts['lstm_num_input']])
        self.maps = tf.placeholder("float", [None, 19, 19, 1])
        self.map_logits = self.tc_model.map_net(self.maps, reuse=reuse)
        self.Inputs = tf.concat((self.X_input, self.map_logits), axis=2)
        self.logits, _ = self.tc_model.net(self.Inputs, reuse=reuse)

        variables_to_restore = [var for var in tf.global_variables() if
                                (var.name.startswith('tclstm') or var.name.startswith('mapnet'))]
        saver = tf.train.Saver(var_list=variables_to_restore)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        if self.mu_model_checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(os.path.join(cur_path, './meta_updater', model_dir))
        else:
            checkpoint = cur_path + './meta_updater/' + model_dir + '/lstm_model.ckpt-' + str(self.mu_model_checkpoint)
        saver.restore(self.sess, checkpoint)

    def metric_init(self, im, init_box, metric_model_path):
        self.metric_model = ft_net(class_num=1120)
        cur_path = os.path.dirname(os.path.abspath(__file__))
        path = cur_path + '/' + metric_model_path
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(path))
        tmp = np.random.rand(1, 3, 107, 107)
        tmp = (torch.autograd.Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        # get target feature
        self.metric_model(tmp)
        init_box = init_box.reshape((1, 4))
        anchor_region = me_extract_regions(im, init_box)
        anchor_region = process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (torch.autograd.Variable(anchor_region)).type(torch.FloatTensor).cuda()
        self.anchor_feature, _ = self.metric_model(anchor_region)

    def local_track(self, image):
        _, out = self.track(image)
        state, score_map, test_x, scale_ind, sample_pos, sample_scales, flag, s = out
        score_map = cv2.resize(score_map, (19, 19))
        update_flag = flag not in ['not_found', 'uncertain']
        update = update_flag
        max_score = max(score_map.flatten())
        self.all_map.append(score_map)
        local_state = np.array(state).reshape((1, 4))
        ap_dis = self.metric_eval(image, local_state, self.anchor_feature)
        self.dis_record.append(ap_dis.data.cpu().numpy()[0])
        h = image.shape[0]
        w = image.shape[1]
        self.state_record.append([local_state[0][0] / w, local_state[0][1] / h,
                                  (local_state[0][0] + local_state[0][2]) / w,
                                  (local_state[0][1] + local_state[0][3]) / h])
        self.rv_record.append(max_score)
        if len(self.state_record) >= tcopts['time_steps']:
            dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            rv = np.array(self.rv_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
            map_input = np.array(self.all_map[-tcopts["time_steps"]:])
            map_input = np.reshape(map_input, [tcopts['time_steps'], 1, 19, 19])
            map_input = map_input.transpose((0, 2, 3, 1))
            X_input = np.concatenate((state_tc, rv, dis), axis=1)
            logits = self.sess.run(self.logits,
                                               feed_dict={self.X_input: np.expand_dims(X_input, axis=0),
                                                          self.maps: map_input})
            update = logits[0][0] < logits[0][1]

        hard_negative = (flag == 'hard_negative')
        learning_rate = getattr(settings, 'hard_negative_learning_rate', None) if hard_negative else None

        if update:
            # Get train sample
            train_x = test_x[scale_ind:scale_ind + 1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind, :],
                                             sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind, ...])
        self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
        return state, score_map, update, max_score, ap_dis.data.cpu().numpy()[0]

    def metric_eval(self, im, boxes, anchor_feature):
        box_regions = me_extract_regions(np.array(im), boxes)
        box_regions = process_regions(box_regions)
        box_regions = torch.Tensor(box_regions)
        box_regions = (torch.autograd.Variable(box_regions)).type(torch.FloatTensor).cuda()
        box_features, class_result = self.metric_model(box_regions)

        class_result = torch.softmax(class_result, dim=1)
        ap_dist = torch.norm(anchor_feature - box_features, 2, dim=1).view(-1)
        return ap_dist

    def get_first_state(self):
        return self.score_map, self.score_max

    def locate(self, image):
        im = augmentation.numpy_to_torch(image)
        sample_pos = self.pos.round()
        sample_scales = self.target_scale * settings.scale_factors
        test_x, _ = self.features.sample_patch_multiscale(im, self.pos, sample_scales, self.img_sample_sz)

        # Compute classification scores
        with torch.no_grad():
            scores_raw = self.features.net.classifier.classify(self.target_filter, test_x)

        # Localize the target
        translation_vec, scale_ind, s, flag = localize_target(scores_raw, sample_pos, sample_scales, super)
        return translation_vec, scale_ind, s, flag, sample_pos, sample_scales, test_x

    def tracking(self, image):
        self.i += 1
        output_state, self.score_map, update, score_max, dis = self.local_track(image)
        # gt_err = self.groundtruth[self.i, 2] < 3 or self.groundtruth[self.i, 3] < 3
        # gt_nan = any(np.isnan(self.groundtruth[self.i]))
        # if gt_err:
        #     iou = -1
        # elif gt_nan:
        #     iou = 0
        # else:
        #     iou = compute_iou(self.groundtruth[self.i], local_state1)

        # width = self.last_gt[3] - self.last_gt[1]
        # height = self.last_gt[2] - self.last_gt[0]

        # if self.p.visualization:
        #     show_res(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
        #              groundtruth=self.groundtruth, update=update,
        #              frame_id=self.i, score=max(self.score_map.flatten()))

        # return [float(self.last_gt[1]), float(self.last_gt[0]), float(width),
        #         float(height)], self.score_map, iou, score_max, dis
        out = {'target_bbox': output_state}
        return out, [output_state, self.score_map, update, score_max, dis]