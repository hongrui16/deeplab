import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory, args = None):
        self.directory = directory
        self.args = args

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step, split = ''):
        num = 5 if len(image) > 5 else len(image)
        grid_image = make_grid(image[:num].clone().cpu().data, num, normalize=True, pad_value = 1)
        writer.add_image(f'{split} Image', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:num], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset, args = self.args), 
                                                        num, normalize=False, range=(0, 255), pad_value = 1)
        writer.add_image(f'{split} Prediction', grid_image, global_step)
        
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:num], 1).detach().cpu().numpy(),
                                                       dataset=dataset, args = self.args), 
                                                num, normalize=False, range=(0, 255), pad_value = 1)
        writer.add_image(f'{split} Groundtruth', grid_image, global_step)

        