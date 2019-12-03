# efficientdet.pytorch
unofficial implementation of efficientdet(WIP)

**please see `train_efficientdet.ipynb` for VOC training.**

- supported efficientnet backbone (B0 for now)
- supported resnet backbone
- now working on BiFPN implementation

|*backbone*|*resolution*|*VOCmAP*    |*COCOmAP*|*Inference[ms]*|*model*|
|:------:|:------------:|:----------:|:-------:|:-------------:|:-----:|
|VGG16   |300           |79.5        |         |               |[here](https://github.com/kentaroy47/ObjectDetection.Pytorch/releases/download/ssdvgg200/ssd300_200.pth)|
|resnet18|300           |76.5        |         |               |       |
|resnet50|300           |80.5        |         |               |       |
|resnet101|300           |           |         |               |       |
|EfficientnetB0|300           |            |         |               |       |
|EfficientnetB1|300           |            |         |               |       |
|EfficientnetB2|300           |           |         |               |       |
