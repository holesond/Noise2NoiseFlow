import os
import sys
import inspect

import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir,"../"))
from data_loader.utils import hps_loader
from train_noise_model import init_params
from noise_model import NoiseModel

def load_checkpoint(model, checkpoint_dir, device):
	checkpoint = torch.load(
        checkpoint_dir, map_location=torch.device(device))
	model.load_state_dict(checkpoint['state_dict'])
	return model, checkpoint['epoch_num']

class SRGBNoiseModel():
    def __init__(
            self, x_shape=[32, 32, 3],
            noise_model_path="our_model_2",
            nm_load_epoch=4, seed=None):
        """
        our_model - finetuned network
            - epoch 240 - good balance of validation NLL and KLD (KLD 0.040)
            - (epoch 340 - the best in terms of validation NLL)
        our_model_2 - network trained from scratch only on Basler data
            - epoch 4 - the lowest KLD of 0.024
            - (epoch 360 - the best in terms of validation NLL (KLD 0.057))
        """
        if seed is not None:
            torch.manual_seed(seed)
        script_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
        x_shape = np.array(x_shape)
        x_shape = list(x_shape[[2,0,1]])
        nm_path = os.path.join(
            script_dir, 'experiments', 'basler', noise_model_path)
        hps = hps_loader(os.path.join(nm_path, 'hps.txt'))
        hps.param_inits = init_params()
        #hps.device = 'cpu'
        nm = NoiseModel(
            x_shape, hps.arch, hps.flow_permutation,
            hps.param_inits, hps.lu_decomp, hps.device, hps.raw)
        nm.to(hps.device)
        
        logdir = os.path.join(nm_path, 'saved_models')
        models = sorted(os.listdir(logdir))
        if nm_load_epoch:
            last_epoch = nm_load_epoch
        else:
            last_epoch = str(max([int(i.split("_")[1]) for i in models[1:]]))
        saved_model_file_name = 'epoch_{}_noise_model_net.pth'.format(last_epoch)
        saved_model_file_path = os.path.join(
            logdir, saved_model_file_name)
        
        nm, nm_epoch = load_checkpoint(
            nm, saved_model_file_path, hps.device)
        print('Noise model epoch is {}'.format(nm_epoch))
        self.model = nm
        self.device = hps.device
        self.temp = hps.temp    # 1.0
        self.iso = np.array([[[[100]]]])
        cam = ['IP', 'GP', 'S6', 'N6', 'G4'].index('GP')
        self.cam = np.array([[[[cam]]]])
    
    
    def process(self, clean_image):
        clean_image = clean_image[...,::-1]
        clean_image = clean_image.transpose((2,0,1))
        clean_image = np.array([clean_image])
        with torch.no_grad():
            self.model.eval()
            kwargs = {
                'clean': torch.from_numpy(clean_image).to(torch.float).to(self.device),
                'eps_std': torch.tensor(self.temp, device=self.device),
                'iso': torch.from_numpy(self.iso).to(torch.float).to(self.device),
                'cam': torch.from_numpy(self.cam).to(torch.float).to(self.device)
            }
            noisy_image = self.model.sample(**kwargs)
        noisy_image = noisy_image[0]
        noisy_image = np.array(noisy_image.cpu()).transpose((1, 2, 0))
        noisy_image = noisy_image[...,::-1]
        return noisy_image


def main(fn_in, fn_out):
    img = imageio.imread(fn_in).astype(np.float32)
    x_shape = img.shape
    noise_model = SRGBNoiseModel(x_shape)
    noisy_image = noise_model.process(img)
    imageio.imwrite(fn_out, noisy_image.astype(np.uint8))
    
def cmd():
    if len(sys.argv) != 3:
        print("Usage: python add_noise.py "
            "/input/image.png /output/noisy_image.png")
        return
    fn_in = sys.argv[1]
    fn_out = sys.argv[2]
    main(fn_in, fn_out)

if __name__ == "__main__":
    cmd()
