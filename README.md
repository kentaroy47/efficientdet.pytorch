# efficientdet.pytorch
unofficial implementation of efficientdet

**please see `train_efficientdet.ipynb` for VOC training.**

- supported efficientnet backbone (B0 for now)
- supported resnet backbone.
- supports BiFPN.
- supports VOC and COCO training.

Now training models.. Plz wait.

|*backbone*|*resolution*|*VOCmAP*    |*COCOmAP*|*Inference[ms]*|*model*|
|:------:|:------------:|:----------:|:-------:|:-------------:|:-----:|
|VGG16   |300           |79.5        |         |               |[here](https://github.com/kentaroy47/ObjectDetection.Pytorch/releases/download/ssdvgg200/ssd300_200.pth)|
|resnet18|300           |76.5        |         |               |       |
|resnet50|300           |80.5        |         |               |       |
|resnet101|300           |           |         |               |       |
|EfficientnetB0(wo/BiFPN)|300      |77.0       |         |               |       |
|EfficientnetB0(w/BiFPN) |300      |79.0       |         |               |       |