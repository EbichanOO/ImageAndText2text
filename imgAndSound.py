import chainer
import numpy as np

def run(epoch=100):
    opt = chainer.optimizers.Adam()
    minibatch_size = 10

    from VGG import ImNet
    im_model = ImNet()
    im_model.to_cpu()
    opt.setup(im_model)

    import soundNet
    so_model = soundNet.SoundNet()
    so_model.to_cpu()
    opt.setup(so_model)

    train_iter_i = chainer.iterators.SerialIterator(i_datas, minibatch_size, repeat=False, shuffle=False)
    train_iter_s = chainer.iterators.SerialIterator(s_data, minibatch_size, repeat=False, shuffle=False)

    import core_utils
    while train_iter_i.epoch < epoch:
        image = train_iter_i.next()
        sound = train_iter_s.next()

        outV = im_model(image)
        outS = so_model(sound)

        cos = core_utils.cos_sim_matrix(outV, outS)
        loss = chainer.functions.mean_squared_error(1, cos)
        print("loss = {loss}")

        im_model.cleargrads()
        so_model.cleargrads()
        loss.backward()
        opt.update()
    save_model = ['imNet_params.npz', 'soNet_params.npz']
    chainer.serializers.save_npz(save_model[0], im_model)
    chainer.serializers.save_npz(save_model[1], so_model)