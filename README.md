# efficientdet.pytorch
unofficial implementation of efficientdet

- supported efficientnet backbone (B0 for now)
- supported resnet backbone.
- supports BiFPN.
- supports VOC and COCO training.

## requirements
`pip install --upgrade efficientnet-pytorch`

## PASCAL-VOC training
- see `train_efficientdet.ipynb`
- see `eval.ipynb` for evaluation

## COCO training
- see `train_efficientdet.ipynb` and configure MS-COCO dataset.

## FP16 training
- see `train_fp16_apex.ipynb`

![configs](https://miro.medium.com/max/678/1*hGTQY4XHbAtPch7nLEp12g.png)

![perf](https://miro.medium.com/max/1922/1*QVVzgKBg7P4nbKHcnO1uKQ.png)

Now training models.. Plz wait.

|*backbone*              |*resolution*|*VOCmAP*    |*COCOmAP*|*Inference[ms]*|*model*|
|:------:                |:------------:|:----------:|:-------:|:-------------:|:-----:|
|EfficientnetB0(wo/BiFPN)|512      |77.0       |TBD         |               |       |
|EfficientnetB0(w/BiFPN) |512      |77.2       |TBD         |               |       |
|EfficientnetB2(wo/BiFPN)|768      |TBD       |TBD         |               |       |
|EfficientnetB2(w/BiFPN) |768      |TBD       |TBD         |               |       |
|EfficientnetB4(wo/BiFPN)|1024      |TBD       |TBD         |               |       |
|EfficientnetB4(w/BiFPN) |1024      |TBD       |TBD         |               |       |
|EfficientnetB6(wo/BiFPN)|1408      |TBD       |TBD         |               |       |
|EfficientnetB6(w/BiFPN) |1408      |TBD       |TBD         |               |       |
