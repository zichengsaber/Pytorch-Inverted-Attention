from collections import OrderedDict
from typing import Tuple,List,Dict,Optional
import torch

import torch.nn.functional as F 
from torch import nn,Tensor


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]) : the original feature maps
        names (List[str])
    Returns:
        results (List[Tensor]): the extend set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor],List[str]]:
        pass

class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps.This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will 
            be performed.It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks:Optional[ExtraFPNBlock]=None,
    
    ):
        super().__init__()
        self.inner_blocks=nn.ModuleList()
        self.layer_blocks=nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")

            inner_block_module=nn.Conv2d(in_channels,out_channels,1)
            layer_block_module=nn.Conv2d(out_channels,out_channels,3,padding=1)

            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
        
        # initialize paramters now to avoid modifying the initialization of 
        # top_blocks
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,a=1)
                nn.init.constant_(m.bias,0)
        
        if extra_blocks is not None:
            assert isinstance(extra_blocks,ExtraFPNBlock)
        self.extra_blocks=extra_blocks
    
    def forward(self,x:Dict[str,Tensor])-> Dict[str,Tensor]:
        """
        Computes the FPN for a set of feature maps

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderDict into two lists for easier handling
        names=list(x.keys())
        x=list(x.values())
        # from top to down
        # the top 
        last_inner=self.inner_blocks[-1](x[-1])
        results=[]
        results.append(self.layer_blocks[-1](last_inner))
        # from top to down
        for idx in range(len(x)-2,-1,-1):
            inner_lateral=self.inner_blocks[idx](x[idx])
            feat_shape=inner_lateral.shape[-2:]
            inner_top_down=F.interpolate(last_inner,size=feat_shape,mode="nearest")
            last_inner=inner_lateral+inner_top_down
            results.insert(0,self.layer_blocks[idx](last_inner))
        
        if self.extra_blocks is not None:
            results,names=self.extra_blocks(results,x,names)
        
        # make it back an OrderedDict
        out=OrderedDict([(k,v) for k,v in zip(names,results)])

        return out 
class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """
    def forward(
        self,
        x:List[Tensor],
        y:List[Tensor],
        names:List[str],
    )->Tuple[List[Tensor],List[str]]:
        names.append("pool")
        # kernel_size,stride,padding
        x.append(F.max_pool2d(x[-1],1,2,0))
        return x,names

class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers,
    P6 and P7
    """
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.p6=nn.Conv2d(in_channels,out_channels,3,2,1)
        self.p7=nn.Conv2d(out_channels,out_channels,3,2,1)
        for module in [self.p6,self.p7]:
            nn.init.kaiming_uniform_(module.weight,a=1)
            nn.init.constant_(module.bias,0)
        
        self.use_P5=in_channels == out_channels
    
    def forward(
        self,
        p:List[Tensor], # res
        c:List[Tensor], # x
        names:List[str],
    )-> Tuple[List[Tensor],List[str]]:
        p5,c5=p[-1],c[-1]
        x=p5 if self.use_P5 else c5 
        p6=self.p6(x)
        p7=self.p7(F.relu(p6))
        p.extend([p6,p7])
        names.extend(["p6","p7"])
        return p, names


if __name__=="__main__":
    m=FeaturePyramidNetwork([10,20,30],5)
    x=OrderedDict()
    x['feat0']=torch.rand(1,10,64,64)
    x['feat2']=torch.rand(1,20,16,16)
    x['feat3']=torch.rand(1,30,8,8)
    output=m(x)
    print([(k,v.size()) for k,v in output.items()])



