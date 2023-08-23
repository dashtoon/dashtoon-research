from adapter import T2IAdapter, CoAdapterFuser
import torch
import torch.nn as nn
from rich.console import Console
import requests
from io import BytesIO
from torchvision import transforms
from PIL import Image
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class Capsule_CoAdapter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # will hold all the adapters
        self.adapters = nn.ModuleList(args)
        
        
        self.fuser = kwargs['coadapter_fuser']
        self.order = kwargs['adapter_order']

    def forward(self, *args):
        
        self.features = dict()
        for k in range(len(self.adapters)):
            feat = self.adapters[k](args[k])
            self.features[self.order[k]] = feat
        # fuse all the features
        out_fused = self.fuser(self.features)
        return out_fused
    
    def get_adapter_features(self):
        return self.features




if __name__ == '__main__':
    adapter1 = T2IAdapter(channels=[320,640,1280], in_channels=3)
    adapter2 = T2IAdapter(channels=[320,640,1280], in_channels=3)
    
    coadapter_fuser = CoAdapterFuser()
    
    fuser = Capsule_CoAdapter(adapter1, adapter2, coadapter_fuser=coadapter_fuser, adapter_order=['sketch','color']).to('cuda')
    
    
    # load image
    url = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main/images/pose.png"

    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))


    # define transforms
    trf = transforms.Compose([
                                transforms.Resize(1024),
                                transforms.ToTensor(),  
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
                                ])

    img = trf(img).unsqueeze(0)
    
    img_batch = torch.concat([img,img]).to('cuda')

    print(img_batch.shape)

    # send [n, c, h, w] of img_batch for adapter1 and [n, c, h, w] of img_batch for adapter2
    out = fuser(img_batch, img_batch)
    
    print([k.shape for k in out])
    
    print([[j.shape for j in k] for k in fuser.get_adapter_features().values()])
    

