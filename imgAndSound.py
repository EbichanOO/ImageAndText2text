import chainer
import numpy as np

def run(epoch=100):
    opt = chainer.optimizers.Adam()
    minibatch_size = 10

    from imageNet import ImNet
    im_model = ImNet()
    im_model.to_cpu()
    opt.setup(im_model)

    import soundNet
    so_model = soundNet.SoundNet()
    so_model.to_cpu()
    opt.setup(so_model)

    import core_utils
    from data_maker import DataMaker
    DTMK = DataMaker()
    sound_iter = DTMK.soundData()
    for video in DTMK.videoData():
        image = video
        try:
            sound = sound_iter.next()
        except:
            sound = np.zeros(100,)

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

if __name__ == '__main__':
    run()