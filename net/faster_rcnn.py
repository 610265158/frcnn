#-*-coding:utf-8-*-
import tensorflow as tf


from train_config import config as cfg


import net.model_frcnn as model_frcnn

from net.model_cascade import CascadeRCNNHead
from net.basemodel import image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
from net.model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses
from net.data import get_all_anchors, get_all_anchors_fpn
from net.model_box import RPNAnchors, clip_boxes, crop_and_resize, roi_align
from net.model_rpn import generate_rpn_proposals, rpn_head, rpn_losses
from net.model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs, fastrcnn_predictions, sample_fast_rcnn_targets




import tensorflow.contrib.slim as slim
def fasterrcnn_arg_scope(weight_decay=0.00001,
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': True,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d,slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params,
      data_format='NCHW'):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME',data_formate='NCHW') as arg_sc:
        return arg_sc






def backbone(image):
    from net.resnet.backbone import plain_resnet50_backbone
    p23456=plain_resnet50_backbone(image)
    return p23456


def backbone_(image):

    c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
    p23456 = fpn_model('fpn', c2345)
    return p23456


def slice_feature_and_anchors( p23456, anchors):
    for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
        with tf.name_scope('FPN_slice_lvl{}'.format(i)):
            anchors[i] = anchors[i].narrow_to(p23456[i])

def rpn(image, features, inputs,L2_reg,training):
    assert len(cfg.RPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

    image_shape2d = tf.shape(image)[2:]     # h,w
    all_anchors_fpn = get_all_anchors_fpn()


    ###get itstower name
    tower_str=None
    for k_,v_ in inputs.items():

        if 'anchor' in k_:
            tower_str=k_.split('/')[0]
            break

    multilevel_anchors = [RPNAnchors(
        all_anchors_fpn[i],
        inputs[tower_str+'/anchor_labels_lvl{}'.format(i + 2)],
        inputs[tower_str+'/anchor_boxes_lvl{}'.format(i + 2)]) for i in range(len(all_anchors_fpn))]


    slice_feature_and_anchors(features, multilevel_anchors)

    # Multi-Level RPN Proposals
    rpn_outputs = [rpn_head('rpn%d'%i, pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS),L2_reg,training)
                   for i,pi in enumerate(features)]

    multilevel_label_logits = [k[0] for k in rpn_outputs]
    multilevel_box_logits = [k[1] for k in rpn_outputs]
    multilevel_pred_boxes = [anchor.decode_logits(logits)
                             for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)]



    proposal_boxes, proposal_scores = generate_fpn_proposals(
        multilevel_pred_boxes, multilevel_label_logits, image_shape2d,L2_reg,training)

    #print('multilevel_pred_boxes',multilevel_pred_boxes)

    if training:
        losses = multilevel_rpn_losses(
            multilevel_anchors, multilevel_label_logits, multilevel_box_logits)
    else:
        losses = []

    return BoxProposals(proposal_boxes), losses

def roi_heads( image, features, proposals, targets,L2_reg,training):
    image_shape2d = tf.shape(image)[2:]     # h,w
    assert len(features) == 5, "Features have to be P23456!"
    gt_boxes, gt_labels, *_ = targets

    if training:
        proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)


    fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)

    if not cfg.FPN.CASCADE:
        roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposals.boxes, 7)


        head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn,L2_reg,training)




        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
            'fastrcnn/outputs', head_feature, cfg.DATA.NUM_CLASS,L2_reg,training)


        fastrcnn_head = FastRCNNHead(proposals, fastrcnn_box_logits, fastrcnn_label_logits,
                                     gt_boxes, tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))


    else:
        def roi_func(boxes):
            return multilevel_roi_align(features[:4], boxes, 7)

        fastrcnn_head = CascadeRCNNHead(
            proposals, roi_func, fastrcnn_head_func,
            (gt_boxes, gt_labels), image_shape2d, cfg.DATA.NUM_CLASS)

    if training:
        all_losses = fastrcnn_head.losses()

        # if cfg.MODE_MASK:
        #     gt_masks = targets[2]
        #     # maskrcnn loss
        #     roi_feature_maskrcnn = multilevel_roi_align(
        #         features[:4], proposals.fg_boxes(), 14,
        #         name_scope='multilevel_roi_align_mask')
        #     maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
        #     mask_logits = maskrcnn_head_func(
        #         'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
        #
        #     target_masks_for_fg = crop_and_resize(
        #         tf.expand_dims(gt_masks, 1),
        #         proposals.fg_boxes(),
        #         proposals.fg_inds_wrt_gt, 28,
        #         pad_border=False)  # fg x 1x28x28
        #     target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
        #     all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))


                ###add predict nms here
        decoded_boxes = fastrcnn_head.decoded_output_boxes()
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
        label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
        final_boxes, final_scores, final_labels = fastrcnn_predictions(
            decoded_boxes, label_scores, name_scope='output')

        final_boxes=tf.identity(final_boxes,name='boxes')
        final_scores = tf.identity(final_scores, name='scores')

        return all_losses
    else:
        decoded_boxes = fastrcnn_head.decoded_output_boxes()
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
        label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
        final_boxes, final_scores, final_labels = fastrcnn_predictions(
            decoded_boxes, label_scores, name_scope='output')
        # if cfg.MODE_MASK:
        #     # Cascade inference needs roi transform with refined boxes.
        #     roi_feature_maskrcnn = multilevel_roi_align(features[:4], final_boxes, 14)
        #     maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
        #     mask_logits = maskrcnn_head_func(
        #         'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
        #     indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
        #     final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
        #     tf.sigmoid(final_mask_logits, name='output/masks')
        return []


def preprocess( image):
    image = tf.expand_dims(image, 0)
    #image = image_preprocess(image, bgr=True)
    return tf.transpose(image, [0, 3, 1, 2])
def faster_rcnn( inputs,L2_reg=0.00001,training=True):
    from net.config import finalize_configs
    finalize_configs(True)

    anchor_inputs = {k: v for k, v in inputs.items() if 'anchor_' in k}

    image = preprocess(inputs['images'])  # 1CHW
    features = backbone(image)

    proposals, rpn_losses = rpn(image, features, anchor_inputs,L2_reg,training)  # inputs?

    targets = [inputs[k] for k in ['gt_boxes', 'gt_labels', 'gt_masks'] if k in inputs]
    head_losses = roi_heads(image, features, proposals, targets,L2_reg,training)
    total_cost=rpn_losses+ head_losses
    return total_cost
