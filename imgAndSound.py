import chainer
import numpy as np

def run(epoch=100):
    opt = chainer.optimizers.Adam()
    minibatch_size = 10

    from VGG import VGG13
    im_model = VGG13()
    im_model.to_cpu()
    opt.setup(im_model)

    from in_sound import change_feature_v

    from bertChainer import modeling
    vi_to_v_model = modeling.BertModel(14925759)
    vi_to_v_model.to_cpu()
    opt.setup(vi_to_v_model)