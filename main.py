def run(epoch=100):
    import chainer
    from chainer.dataset import convert
    import numpy as np
    opt = chainer.optimizers.Adam()
    minibatch_size = 10

    from VGG import VGG13
    visual_model = VGG13()
    visual_model.to_cpu()
    opt.setup(visual_model)

    from bertChainer import modeling
    sentence_model = modeling.BertModel(14925759)
    sentence_model.to_cpu()
    opt.setup(sentence_model)
    s_datas = [int.from_bytes(j.encode('utf-8'), 'big') for j in x]

    train_iter_v = chainer.iterators.SerialIterator(v_datas, minibatch_size, repeat=False, shuffle=False)
    train_iter_s = chainer.iterators.SerialIterator(s_datas, minibatch_size, repeat=False, shuffle=False)

    import core_utils
    UNIST = core_utils.understand()
    while train_iter_v.epoch < epoch:
        images = train_iter_v.next()
        v_out = visual_model(images)
        sentences = train_iter_s.next()
        s_out = sentence_model(sentences)
        
        cosV = core_utils.cos_sim_matrix(v_out, s_out)
        loss = chainer.functions.mean_squared_error(1, cosV)
        
        visual_model.cleargrads()
        loss.backward()
        opt.update()

        sentence_model.cleargrads()
        loss.backward()
        opt.update()

        UNIST.update(s_out, v_out)
    save_model = ['vgg_model.npz', 'bert_model.npz']
    chainer.serializers.save_npz(save_model[0], opt)
    chainer.serializers.save_npz(save_model[1], opt)
    UNIST.saveSpace()

