import time

import torch
import torch.nn as nn

from .loss_multi import lossAV
from .loconet_encoder import locoencoder


class Loconet(nn.Module):

    def __init__(self, cfg):
        super(Loconet, self).__init__()
        self.cfg = cfg
        self.model = locoencoder(cfg)
        self.lossAV = lossAV()
        
    def forward_visual_frontend(self, visualFeature, **kwargs):
        return self.model.forward_visual_frontend(visualFeature)
    
    def forward(self, audioFeature, visualFeature, **kwargs):
        # input(f"visualFeature.shape : {visualFeature.shape}")
        
        b, s, t = visualFeature.shape[0], visualFeature.shape[1], visualFeature.shape[2]
        visualFeature = visualFeature.view(b * s, *visualFeature.shape[2:])
        
        audioEmbed = self.model.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature, **kwargs)
        audioEmbed = audioEmbed.repeat(s, 1, 1)
        
        # print(f"audioEmbed.shape : {audioEmbed.shape} visualEmbed.shape : {visualEmbed.shape}")
        audioEmbed, visualEmbed = self.model.forward_cross_attention(
            audioEmbed, visualEmbed)
        outsAV = self.model.forward_audio_visual_backend(audioEmbed, visualEmbed, b, s)
        outsAV = outsAV.view(b, s, t, -1)[:, 0, :, :].view(b * t, -1)
        predScore = self.lossAV.forward(outsAV)
        
        return predScore

    
class Laser(Loconet):

    def __init__(self, cfg, n_channel, layer):
        super(Laser, self).__init__(cfg)

        self.n_channel = n_channel
        self.landmark_bottleneck = nn.Conv2d(in_channels=164, out_channels=n_channel, kernel_size=(1, 1))

        # insert before layer
        self.layer = layer

        if layer == 0:
            self.bottle_neck = nn.Conv2d(in_channels=(1 + n_channel), out_channels=1, kernel_size=(1, 1))
        elif layer == 1:
            self.bottle_neck = nn.Conv2d(in_channels=(64 + n_channel), out_channels=64, kernel_size=(1, 1))
        elif layer == 2:
            self.bottle_neck = nn.Conv2d(in_channels=(64 + n_channel), out_channels=64, kernel_size=(1, 1))
        elif layer == 3:
            self.bottle_neck = nn.Conv2d(in_channels=(128 + n_channel), out_channels=128, kernel_size=(1, 1))
        elif layer == 4:
            self.bottle_neck = nn.Conv2d(in_channels=(256 + n_channel), out_channels=256, kernel_size=(1, 1))
            

    def create_landmark_tensor(self, landmark, dtype, device):
        """
            landmark has shape (b, s, t, 82, 2)
            return tensor has shape (b, s, t, 164, W, H)
        """
        landmarkTensor = None
        
        if self.layer == 0:
            W, H = 112, 112
        elif self.layer == 1:
            W, H = 28, 28
        elif self.layer == 2:
            W, H = 28, 28
        elif self.layer == 3:
            W, H = 14, 14
        elif self.layer == 4:
            W, H = 7, 7

        b, s, t, _, _ = landmark.shape
        landmarkTensor = torch.zeros((b, s, t, 82, 2, W, H), dtype=dtype, device=device)
        landmark_idx = ((landmark > 0.0) | torch.isclose(landmark, torch.tensor(0.0))) & ((landmark < 1.0) | landmark.isclose(landmark, torch.tensor(1.0)))

        landmark_masked = torch.where(landmark_idx, landmark, torch.tensor(float('nan')))

        coordinate = torch.where(torch.isnan(landmark_masked), torch.tensor(float('nan')), torch.min(torch.floor(landmark_masked * W), torch.tensor(W - 1)))

        # Convert coordinates to long, handling NaN to avoid indexing issues
        coord_0 = coordinate[..., 0].long()
        coord_1 = coordinate[..., 1].long()
        
        # Create a mask for valid coordinates (non-NaN)
        valid_mask = ~torch.isnan(coordinate[..., 0]) & ~torch.isnan(coordinate[..., 1])

        # Get valid indices
        b_id, s_id, t_id, lip_id = torch.nonzero(valid_mask, as_tuple=True)

        if b_id.numel() > 0:  # Ensure there are valid indices
            landmarkTensor[b_id, s_id, t_id, lip_id, :, coord_0[b_id, s_id, t_id, lip_id], coord_1[b_id, s_id, t_id, lip_id]] = landmark[b_id, s_id, t_id, lip_id, :]

        landmarkTensor = landmarkTensor.reshape(b * s * t, -1, W, H)

        assert (landmarkTensor.shape[1] == 164)
        return landmarkTensor

     # hook the gradient of the activation
    def activations_hook(self, grad):
        self.gradients = grad

    def forward_visual_frontend_on_emb(self, x, landmarkFeature):
        B, T, W, H = x.shape
        if self.layer == 0:
            x = x.view(B * T, 1, W, H)
            x = torch.cat((x, landmarkFeature), dim = 1)
            x = self.bottle_neck(x)
            x = x.view(B, T, W, H)
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = x.transpose(0, 1).transpose(1, 2)
        batchsize = x.shape[0]
        x = self.model.visualFrontend.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3],
                              x.shape[4]) # b * s * t, c, w, h
        
        # inject before self.layer
        # landmarkFeature has shape (b * s * t, n_channel, H, W)
        # x has shape (b * s * t, c, W, H)

        layers = [self.model.visualFrontend.resnet.layer1, self.model.visualFrontend.resnet.layer2, self.model.visualFrontend.resnet.layer3, self.model.visualFrontend.resnet.layer4]
        for i in range(4):
            if i == self.layer - 1:
                x = torch.cat((x, landmarkFeature), dim = 1)
                x = self.bottle_neck(x)
                x = layers[i](x)
            else:
                x = layers[i](x)
            if i == 2:
                # hook the gradient
                if x.requires_grad:
                    x.register_hook(self.activations_hook)

        x = self.model.visualFrontend.resnet.avgpool(x)
        x = x.reshape(batchsize, -1, 512)
        x = x.transpose(1, 2)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.model.visualTCN(x)
        x = self.model.visualConv1D(x)
        x = x.transpose(1, 2)
        return x
    
    def forward_visual_frontend(self, visualFeature, **kwargs):
        
        landmarkFeature = self.create_landmark_tensor(kwargs["landmark"], visualFeature.dtype, visualFeature.device)
        landmarkFeature = self.landmark_bottleneck(landmarkFeature, landmarkFeature)
        
        return self.forward_visual_frontend_on_emb(visualFeature)


class loconet(nn.Module):

    def __init__(self, cfg, *args):
        super(loconet, self).__init__()
        self.cfg = cfg
        args = (4, 1) if cfg.USE_LASER_LOCONET else ()
        model_name = "Laser" if cfg.USE_LASER_LOCONET else "Loconet"
        self.model = globals()[model_name](cfg, *args)

        print(
            time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" %
            (sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location='cpu')
        
        new_state = {}
        for k, v in loadedState.items():
            new_state[k.replace("model.module.", "model.")] = v
        info = self.load_state_dict(new_state, strict=False)

        print(info)
