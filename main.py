import chainer
opt = chainer.optimizers.Adam()

from VGG import VGG13

visual_model = VGG13()
visual_model.to_cpu()
opt.setup(visual_model)

from bertChainer import modeling
sentence_model = modeling.BertModel(14925759)
sentence_model.to_cpu()
opt.setup(sentence_model)

train_iter_v = chainer.iterators.SerialIterator(v_datas, minibatch_size)
train_iter_s = chainer.iterators.SerialIterator(s_datas, minibatch_size)

epoch = 100
while train_iter.epoch < epoch:
    images = train_iter.next()
    visual_out = VGG13(images)
    loss = chainer.functions.mean_squared_error(1, cosV)