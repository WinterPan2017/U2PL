# 准备
数据集Pascal VOC
1. 解压VOCtrainval_11-May-2012.tar, SegmentationClassAug.zip
2. 将SegmentationClassAug文件夹移动到VOC2012中, 文件目录如下所示:
```
-VOC2012
  |-JPEGImages
  |-SegmentationClass
  |-SegmentationClassAug
  ...
```
3. 软链接VOC2012到data目录
```
ln -s path/to/VOC2012 path/to/U2PL/data
```
4. 修改pretrain checkpoint路径(u2pl/models/resnet.py#20)
5. 依赖安装
```
pip install -r requirements.txt
```

# RUN
```
cd experiments/pascal/1464/lcr
bash run.sh 4 1234
```
log和checkpoin会保存在experiments/pascal/1464/lcr下