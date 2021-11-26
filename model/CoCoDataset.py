import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import transforms

class COCODataset(Dataset):
    def __init__(self,data_dir,split,train=False):
        super().__init__()

        self.data_dir=data_dir 
        self.split=split
        self.train=train

        ann_file=osp.join(data_dir,"annotations/instances_{}.json".format(split))
        self.coco=COCO(ann_file)
        self.ids=[str(k) for k in self.coco.imgs] # str(id)
        # class Ids start from 1
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
    
    def get_image(self,img_id): # str
        img_id=int(img_id)
        img_info=self.coco.imgs[img_id]
        image=Image.open(osp.join(self.data_dir,f"{self.split}",img_info["file_name"])).convert("RGB")
        return image

    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
    
    def get_target(self,img_id):
        img_id=int(img_id)
        ann_ids=self.coco.getAnnIds(img_id)
        anns=self.coco.loadAnns(ann_ids)
        boxes=[]
        labels=[]
        masks=[]

        if len(anns)>0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask=self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
            
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
        
        target=dict(image_id=torch.tensor([img_id]),boxes=boxes,labels=labels,masks=masks)

        return target
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id=self.ids[index]
        image=self.get_image(img_id)
        image=transforms.ToTensor()(image) # 经过归一化后
        target=self.get_target(img_id) if self.train else {}
        return image,target
    


if __name__=="__main__":
    PATH="/home/ZhangZicheng/ObjectionDetection/data/mscoco2017"
    cocodata=COCODataset(PATH,"train2017",True)
    from torch.utils.data import DataLoader
    dataloader=DataLoader(cocodata,batch_size=1,shuffle=True,num_workers=2)
    print(len(cocodata))
    print(len(dataloader))
    for i,(image,target) in enumerate(dataloader):
        print(image.size())
        print(target["boxes"].size())
        print(target["labels"].size())
        print(target["masks"].size())
        break
