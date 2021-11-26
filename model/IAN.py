import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FasterRCNN 
from torchvision.models.detection.roi_heads import RoIHeads
import torch
from torch import nn, Tensor
from typing import Tuple, List, Dict, Optional
from CoCoDataset import COCODataset

from faster_rcnn import fasterrcnn_resnet50_fpn


class ROIHeadsWithIAN():
    def __init__(self):
        self.gradients:Tensor=None
        self.invertedAttNet=IAN(
            spatial_threshold=1e-7,
            channel_threshold=1e-6,
        )
        
    def save_gradients(self,grad):
        self.gradients=grad

    def __call__(self,
                 model:RoIHeads,
                 features:Dict[str,Tensor],
                 proposals:List[Tensor], 
                 image_shapes:List[Tuple[int,int]],
                 targets:Optional[List[Dict[str,Tensor]]]=None):
        if model.training:
            proposals, matched_idxs, labels, regression_targets = model.select_training_samples(proposals, targets)
        else:
            labels=None
            regression_targets=None
            matched_idxs=None
        # 获得ROI_pool 之后的box_features
        box_features=model.box_roi_pool(features, proposals, image_shapes)
        box_features.register_hook(self.save_gradients)
        # 获取box_features的grad
        mlp_features=model.box_head(box_features)
        cls_features=model.box_predictor.cls_score(mlp_features)
        indices=torch.arange(cls_features.size(0),dtype=torch.int64)
        labels_cat=torch.cat(labels)
        final_features=cls_features[indices,labels_cat[indices]]
        total=torch.sum(final_features)/cls_features.size(0) # sum(tensor:[N,])/N
        total.backward()
        # model.zero_grad()
        inverted_attention_map=self.invertedAttNet(box_features,self.gradients)

        return inverted_attention_map
        
        




    


# 不需要产生梯度
class IAN():
    def __init__(self,spatial_threshold,channel_threshold):
        self.Ts=spatial_threshold
        self.Tc=channel_threshold
        self.pool=nn.AdaptiveAvgPool2d((1,1))
    
    def __call__(self,
                 features:Tensor,
                 gradients:Tensor):
        N,C,H,W=features.size()
        feat=features.clone().detach() #[N,C,H,W]
        grad=gradients.clone().detach() #[N,C,H,W]
        pooled_grad=self.pool(grad) # [N,C,1,1]
        attention_map=pooled_grad*feat # [N,C,H,W]
        # spatial Thresh
        inverted_attention_map=torch.ones_like(attention_map) # [[1,1..1]]
        inverted_attention_map[attention_map>self.Ts]=0
        # channel Thresh
        boolen_grad=(pooled_grad<=self.Tc).repeat(1,1,H,W)
        inverted_attention_map[boolen_grad]=1
        
        return inverted_attention_map.requires_grad_(True)


class FasterRCNNwithIAN(nn.Module):
    def __init__(self,model:FasterRCNN):
        super().__init__()
        self.transform=model.transform
        self.backbone=model.backbone
        self.rpn=model.rpn
        self.roi_heads=model.roi_heads
        self.roi_heads_IAN=ROIHeadsWithIAN()
       
        
    def forward(self,
                images:List[Tensor],
                targets:Optional[List[Dict[str,Tensor]]]=None
    )->Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        # 获取原图像大小
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        

        images, targets = self.transform(images, targets)

        features=self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        # 只在training的过程中使用IAN
        """
        if self.training:
            invert_attmap=self.roi_heads_IAN(self.roi_heads,features, proposals, images.image_sizes, targets)
            detections, detector_losses = self.roi_heads(features ,proposals, images.image_sizes, targets,invert_attmap)
        else:
        """
        detections, detector_losses = self.roi_heads(features ,proposals, images.image_sizes, targets)
        
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)


        if self.training:
            return losses
        return detections
         




# test
if __name__=="__main__":
    # prepare data
    cocodataset=COCODataset("../data/mscoco2017","train2017",True)
    image1,target1=cocodataset[3]
    image2,target2=cocodataset[5]
    images=[image1,image2]
    targets=[target1,target2]
    # model
    model=fasterrcnn_resnet50_fpn(pretrained=True)
    iAN=FasterRCNNwithIAN(model)
    print(iAN)
    # 模拟在一个batch中的训练情况
    # test forward
    losses=model(images,targets)
    print(losses)
    sum=0
    for k,v in losses.items():
        sum+=v
    # test backward
    sum.backward()
    

    
    
