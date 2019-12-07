# efficientdet.pytorch
unofficial implementation of efficientdet

## requirements
`pip install --upgrade efficientnet-pytorch`

**please see `train_efficientdet.ipynb` for VOC training.**

- supported efficientnet backbone (B0 for now)
- supported resnet backbone.
- supports BiFPN.
- supports VOC and COCO training.

![configs](https://miro.medium.com/max/678/1*hGTQY4XHbAtPch7nLEp12g.png)

![perf](https://miro.medium.com/max/1922/1*QVVzgKBg7P4nbKHcnO1uKQ.png)

Now training models.. Plz wait.

|*backbone*              |*resolution*|*VOCmAP*    |*COCOmAP*|*Inference[ms]*|*model*|
|:------:                |:------------:|:----------:|:-------:|:-------------:|:-----:|
|EfficientnetB0(wo/BiFPN)|512      |77.0       |TBD         |               |       |
|EfficientnetB0(w/BiFPN) |512      |79.0       |TBD         |               |       |
|EfficientnetB2(wo/BiFPN)|768      |TBD       |TBD         |               |       |
|EfficientnetB2(w/BiFPN) |768      |TBD       |TBD         |               |       |
|EfficientnetB4(wo/BiFPN)|1024      |TBD       |TBD         |               |       |
|EfficientnetB4(w/BiFPN) |1024      |TBD       |TBD         |               |       |
|EfficientnetB6(wo/BiFPN)|1408      |TBD       |TBD         |               |       |
|EfficientnetB6(w/BiFPN) |1408      |TBD       |TBD         |               |       |
