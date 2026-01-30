## 数据集精度测试说明(可选)
### 数据集下载
https://image-net.org/download.php 
### rknn3板端测试
将下好的数据解压并推到板子上，例如/data/imagenet目录下，需包含下面文件：  
> root@rk3588-buildroot:/data/imagenet# ls        
ILSVRC2012_img_val_256	ILSVRC2012_img_val_256.txt
     
将编译好的文件和模型推到板子上，执行      
测试mobilenet_v1
> ./dataset_eval model/mobilenet_v1.rknn model/mobilenet_v1.weight /data/imagenet/ 0x01
    
### python测试
参考python/dataset_eval.py，修改数据集路径和模型路径即可    

