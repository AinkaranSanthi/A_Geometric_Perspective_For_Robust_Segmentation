import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from loss import DiceLoss, FocalLoss
from einops import rearrange
import nibabel as nib
from collections import defaultdict
import torchio as tio
import torch.distributed as dist
import torchvision
from randconv import randconv


def weights_init(m):
    classname = m.__class__.__name__
    if  isinstance(m,  nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif  isinstance(m,  nn.Conv3d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif  isinstance(m,  nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif  isinstance(m,  nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class TrainSE3:
    def __init__(self, args, model):
        self._model = model#.to(device = device)
        self.device = args.device
        self.adc_img = args.adc_img
        self.lr = args.lr
        self.wd = args.weight_decay
        self.image_format = args.image_format
        self.output_dir = args.output_dir
        self.num_classes = args.classes
        self.patch_size = args.patch_size
        self.epochs = args.epochs
        self.dim = args.dim
        self.image_size = args.image_size
        self.topo_fg_only = args.topo_fg_only
        self.opt_vq = self.configure_optimizers(args.beta1, args.beta2)
        self.prepare_training(args.output_dir)
 

    def configure_optimizers(self, beta1, beta2):
        lr = self.lr
        opt_vq = torch.optim.AdamW(
            self._model.parameters(),
            lr=lr, eps=1e-08, betas=(beta1, beta2) )

        return opt_vq


    def save_images(self, outputs, imagespath, image_format, epoch, batch_idx, image_type):
        for i in range(outputs.shape[0]):
            output = torch.squeeze(outputs[i])
            directory = os.path.join(imagespath,'image{img}batch{batch}'.format(img = i,batch=batch_idx))
            os.makedirs(directory, exist_ok = True)
            filename =  os.path.join(directory,'{image_t}epoch{epoch}.{filetype}'.format(image_t=image_type,epoch=epoch, filetype = 'nii'))# if self.image_format == 'nifti' else 'png'))
            if image_format == 'nifti':
               output = output.cpu().detach().numpy().astype(np.float32)
               affine = np.eye(4)
               output = nib.Nifti1Image(output, affine)
               nib.save(output, filename)
            else:
               torchvision.utils.save_image(output, filename) 
                
        return None
    @staticmethod
    def prepare_training(output_dir):
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok = True)
    
    def _all_reduce(self, x, wr):
        # average x over all GPUs
        dist.all_reduce(x, dist.ReduceOp.SUM)
        return x / wr
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
     
    def train(self, train_dataset, val_dataset):
        steps_per_epoch = len(train_dataset)
        min_loss = np.inf
        min_dice = np.inf
        writer = SummaryWriter()
        dice_loss = DiceLoss(self.num_classes)
        patch_dim = [int(self.image_size[i]/self.patch_size[i]) for i in range(len(self.image_size))]

        for epoch in range(self.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for k , imgs in zip(pbar, train_dataset):
                    vqloss = 0
                    invariantloss = 0
                    diceloss = 0
                    if self.adc_img:
                       t2img, t2lab, adcimage = imgs['t2image'].to(device=self.device,  dtype=torch.float), imgs['t2labels'].to(device=self.device,  dtype=torch.long), imgs['adcimage'].to(device=self.device,  dtype=torch.float)
                       img = torch.cat((t2img, adcimage), dim = 1)
                       lab = t2lab.as_tensor()
                    else:
                       img, lab= imgs['t2image'].to(device=self.device,  dtype=torch.float), imgs['t2labels'].to(device=self.device,  dtype=torch.long)
                       img = randconv(img, 1, True, 0.75, self.dim)
                       lab = lab.as_tensor()
                    out, latentt2, latentadc, vqt2, q_loss = self._model(img)
                    out = torch.softmax(out, dim = 1)
                    outt = out[:,1:] if self.topo_fg_only == False else torch.unsqueeze(torch.sum(out[:,1:],dim=1), dim =1)
                    vqloss += q_loss
                    diceloss += dice_loss(out, lab, softmax=False)
                    invariantloss += torch.sum((latentt2 - latentadc)**2) if self.adc_img == True else 0
                    labt = lab[:,1:] if self.topo_fg_only == False else torch.unsqueeze(torch.sum(lab[:,1:],dim=1), dim =1)

                    total_loss = diceloss + vqloss + invariantloss
                    print('epoch:%.3f'% (epoch), 'seg: %.3f' % (diceloss), 'qloss: %.3f' % (vqloss), 'invariantloss: %.3f' % (invariantloss))
                    writer.add_scalar('VQLoss/train', total_loss,  k*epoch)
                    writer.add_scalar('seg_loss/train', diceloss,  k*epoch)
                    self.opt_vq.zero_grad()
                    total_loss.backward()
                    self.opt_vq.step()
         
                    pbar.set_postfix(
                        total_Loss=np.round(total_loss.cpu().detach().numpy().item(), 5),
                        dice=np.round(diceloss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                checkpoint_path = os.path.join(self.output_dir, "checkpoints")
                val_dice, val_loss = self.val(val_dataset, epoch)
                torch.save(self._model.state_dict(), os.path.join(checkpoint_path, "sheaftrain.pt")) if val_loss < min_loss  else None
                min_loss = val_loss if val_loss < min_loss else min_loss
                min_dice = val_dice if val_dice < min_dice else min_dice
                print('min_val_dice:%.3f'% (min_dice))
    def val(self,  val_dataset, epoch):
        loss = defaultdict(list)        
        dice_loss = DiceLoss(self.num_classes-1)
        m = nn.Softmax(dim=1)
        lossce = nn.BCELoss()
        writer = SummaryWriter()

        with tqdm(range(len(val_dataset))) as pbar:
                for k, imgs in enumerate(val_dataset):
                    img, lab = imgs['t2image'].to(device=self.device,  dtype=torch.float), imgs['t2labels'].to(device=self.device,  dtype=torch.long)
                    lab = torch.squeeze(self._one_hot_encoder(lab), dim = 2)
                   
                    with torch.no_grad():
                        out, latentt2, latentadc, vqt2, q_loss  = self._model(img)
                        outv = torch.softmax(out, dim = 1)
                        outh = torch.where(outv>0.5, 1, 0)
                        dice = dice_loss(outv[:,1:], lab[:,1:], softmax = False)
                        hd = compute_hausdorff_distance(outh,lab, include_background=False, percentile = 95)
                        val_loss = dice + q_loss
                    loss['dice'].append(dice)
                    loss['val_loss'].append(val_loss) 
                    print('epoch:%.3f'% (epoch), 'val q Loss: %.3f' % (q_loss), 'val dice Loss 1: %.3f' % (dice), 'val hd Loss 1: %.3f' % (torch.mean(hd)))
                    writer.add_scalar('VQLoss/val',val_loss,  k*epoch)
                    writer.add_scalar('seg_loss/val',dice,  k*epoch)
                    image_path = os.path.join(self.output_dir, "sheafval")
                    out = torch.softmax(outv, dim=1)
                    out = torch.argmax(out, dim = 1)
                    lab = torch.argmax(out, dim = 1)
                    self.save_images(out, image_path,  self.image_format,epoch, k, 'seg_image')
                    self.save_images(img, image_path,  self.image_format, epoch, k, 'original_image')
                    self.save_images(lab, image_path,  self.image_format, epoch, k, 'label_image')
        dice_mean = torch.mean(torch.stack(loss['dice'])) if len(loss['dice']) > 1 else loss['dice'][0]
        val_loss_mean = torch.mean(torch.stack(loss['val_loss'])) if len(loss['val_loss']) > 1 else loss['val_loss'][0]
        return dice_mean.cpu().detach().numpy(), val_loss_mean.cpu().detach().numpy()   
    
    def test(self, model,  model_path, test_dataset):
         _model =  model.to(device=self.device)
         _model.load_state_dict(torch.load(model_path))
         writer = SummaryWriter()
         dice_loss = DiceLoss(self.num_classes-1)
         loss = defaultdict(list)

         with tqdm(range(len(test_dataset))) as pbar:
                for k, imgs in enumerate(test_dataset):
                    img, lab = imgs['t2image'].to(device=self.device,  dtype=torch.float), imgs['t2labels'].to(device=self.device,  dtype=torch.long) 
                    lab = torch.squeeze(self._one_hot_encoder(lab), dim = 2)
                    with torch.no_grad():
                        out, latentt2, latentadc, vqt2, q_loss  = self._model(img)
                        outv = torch.softmax(out, dim = 1)
                        outh = torch.where(outv>0.5, 1, 0)
                        dice = dice_loss(outv[:,1:], lab[:,1:], softmax = False)
                        hd = compute_hausdorff_distance(outh,lab, include_background=False, percentile = 95)
                        test_loss = dice + q_loss
                    loss['dice'].append(dice)
                    loss['test_loss'].append(test_loss) 
                    print('test sample:%.3f'% (k), 'test q Loss: %.3f' % (q_loss), 'test dice Loss 1: %.3f' % (dice), 'test hd Loss 1: %.3f' % (torch.mean(hd)))
                    writer.add_scalar('VQLoss/test',test_loss,  k)
                    writer.add_scalar('seg_loss/test',dice,  k)
                    image_path = os.path.join(self.output_dir, "sheaftest")
                    out = torch.softmax(outv, dim=1)
                    out = torch.argmax(out, dim = 1)
                    lab = torch.argmax(out, dim = 1)
                    self.save_images(out, image_path,  self.image_format, self.epochs, k, 'seg_image')
                    self.save_images(img, image_path,  self.image_format, self.epochs, k, 'original_image')
                    self.save_images(lab, image_path,  self.image_format, self.epochs, k, 'label_image')
         dice_mean = torch.mean(torch.stack(loss['dice'])) if len(loss['dice']) > 1 else loss['dice'][0]
         test_loss_mean = torch.mean(torch.stack(loss['test_loss'])) if len(loss['test_loss']) > 1 else loss['test_loss'][0]

         return dice_mean.cpu().detach().numpy(), test_loss_mean.cpu().detach().numpy()
