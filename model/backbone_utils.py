import warnings
from typing import Callable, Dict, Optional, List, Union
from collections import OrderedDict
from torch import nn,Tensor
from torchvision.ops import misc as misc_nn_ops
from fpn import FeaturePyramidNetwork,LastLevelMaxPool,ExtraFPNBlock

from torchvision.models import resnet



class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.


    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name,new_name]): a dict containing the names of modules
        for which the activations will be returned as the key of the dict,
        and the value of the dict is the name of the returned activation 
    """
    def __init__(self,model:nn.Module,return_layers:Dict[str,str])-> None:
        # 检查model名称是否匹配
        """
        print(return_layers)
        {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        """
        if not set(return_layers).issubset([name for name,_ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        """
        eg:{'layer1': 'feat1', 'layer3': 'feat2'}
        """
        orig_return_layers=return_layers
        return_layers={str(k):str(v) for k,v in return_layers.items()}

        layers=OrderedDict()
        # we just need submodule before our query
        for name,module in model.named_children():
            layers[name]=module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super().__init__(layers)
        self.return_layers=orig_return_layers
    
    def forward(self,x):
        out=OrderedDict()
        for name,module in self.items():
            x=module(x)
            if name in self.return_layers:
                out_name=self.return_layers[name]
                out[out_name]=x 
        return out 

class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses `torchvision.models._utils.IntermediateLayerGetter` to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name,new_name]): a dict containing the names of modules
            for which the activations will be returned as the key of the dict,
            and the value of the dict is the name of the returned activation 
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN
    Attributes:
        out_channels (int): the number of channels in FPN

    """
    def __init__(self,
        backbone: nn.Module,
        return_layers: Dict[str,str],
        in_channels_list: List[int],
        out_channels:int,
        extra_blocks:Optional[ExtraFPNBlock]=None,
    )->None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks=LastLevelMaxPool()
        
        self.body=IntermediateLayerGetter(backbone,return_layers)
        self.fpn=FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )

        self.out_channels=out_channels
    
    def forward(self,x:Tensor)-> Dict[str,Tensor]:
        x=self.body(x)
        x=self.fpn(x)
        return x


def resnet_fpn_backbone(
    backbone_name:str,
    pretrained:bool,
    norm_layer:Callable[...,nn.Module]=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers:int=3,
    returned_layers:Optional[List[int]]=None,
    extra_blocks:Optional[ExtraFPNBlock]=None,
)-> BackboneWithFPN:
    """
    Constructs a specified Resnet backbone with FPN on top .
    Freezes the specified number of layers

    Args:
        backbone_name (str): resnet architecture
        pretrained (bool) : If True, returns a model with backbone pre-trained on Imagenet
        norm_layer (callable):it is recommended to use the default value.
        trainable_layers (int): number of trainable (not frozen)
            resnet layers starting from final block,
            Valid values are between 0 and 5, 
            with 5 meaning all backbone layers are trainable.
        returned_layers (List[int]) The layers of the network to return
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default a ``LastLevelMaxPool`` is used.
    """
    backbone=resnet.__dict__[backbone_name](pretrained=pretrained,norm_layer=norm_layer)
    return _resnet_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks)

def _resnet_fpn_extractor(
    backbone: resnet.ResNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
) -> BackboneWithFPN:
    # select layers that won't be frozen
    """The standard arch of Resnets
    conv1
    bn1
    relu
    maxpool
    layer1
    layer2
    layer3
    layer4
    avgpool 
    fc
    
    """
    assert 0<=trainable_layers<=5
    # total five big conv blocks
    layers_to_train=["layer4","layer3","layer2","layer1","conv1"][:trainable_layers]
    if trainable_layers==5:
        layers_to_train.append("bn1")
    
    for name,parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    
    if extra_blocks is None:
        extra_blocks=LastLevelMaxPool()
    
    if returned_layers is None:
        returned_layers=[1,2,3,4]

    assert min(returned_layers) > 0 and max(returned_layers) < 5

    return_layers={f"layer{k}":str(v) for v,k in enumerate(returned_layers)}

    in_channels_stage2=backbone.inplanes // 8 # 256
    in_channels_list=[in_channels_stage2*2**(i-1) for i in returned_layers]
    
    out_channels=256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

def _validate_trainable_layers(
    pretrained:bool,
    trainable_backbone_layers:Optional[int],
    max_value:int,
    default_value:int,
)-> int:
    # don't freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                f"falling back to trainable_backbone_layers={max_value} so that all layers are trainable"
            )
            trainable_backbone_layers=max_value
    # by default freeze first blcoks
    if trainable_backbone_layers is None:
        trainable_backbone_layers=default_value
    assert 0<=trainable_backbone_layers<=max_value
    return trainable_backbone_layers
