# ==========================================================================
#                             combine the generator model
#            from GP-VTON with the T2I adapter model for use with SDXL                                  
# ==========================================================================

from pathlib import Path
from adapter import T2IAdapter
from rich.console  import Console
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
# import sys
# sys.path.insert(0, "/workspace/GP-VTON")
import sys
sys.path.append('.')
from GP_VTON.models import ResUnetGenerator





class Model_generator_adapter(ModelMixin, ConfigMixin):
    """
    generate the sample with generator and extarct features with T2I sdxl adapter    
    """
    
    
    config_name = "generator_adapter"
    @register_to_config
    def __init__ (self, adapter, generator_model):
        super().__init__()
        self.adapter = adapter
        self.generator_model = generator_model
        
        
    
    def forward (self, x):
        self.after_generator = self.generator_model(x)
        self.feat = self.adapter(self.after_generator)
        
        return self.feat
        
    
    def count_parameters(self,):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")


if __name__ == '__main__':
        

    # ==========================================================================
    #                         create the generator_t2i adapter                                  
    # ==========================================================================

    gen_model = ResUnetGenerator(input_nc=3, output_nc=3, num_downs=5, ngf=64, norm_layer=torch.nn.BatchNorm2d)
    adapter = T2IAdapter(channels=[320, 640, 1280])
    gen_adapter_model = Model_generator_adapter(adapter=adapter, generator_model=gen_model).to('cuda')

    gen_adapter_model.count_parameters()
    
    Console().print(f"[cyan]sending input sample to the [red]Generator_Adapter")
    random_input = torch.randn(12, 3, 512, 512).to('cuda')
    Console().print(f"{random_input.shape}", style='red on black')

    features = gen_adapter_model(random_input)

    Console().print(f"output features shapes are:")
    [Console().print(i.shape) for i in features]


    
    # # --------------------------------------------------------------------------
    # #                          saving the model                        
    # # --------------------------------------------------------------------------
    
    torch.save(gen_adapter_model.state_dict(), "generator_adapter/generator_adapter.pt")

