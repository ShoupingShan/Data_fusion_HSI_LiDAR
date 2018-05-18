# Data_fusion_HSI_LiDAR
**HSI and LiDAR image fusion based on Deep Learning**

## Author
[@Shoupingshan](https://github.com/ShoupingShan)

## Platform
  1. Ubuntu 14.04
  2. CUDA 8.0
  3. GTX 850M
  4. tensorflow-1.4
  5. python2/python3

## Architecture of Convolutional Neural Network used
**input- [conv - relu - maxpool] x 2 - [affine - relu] x 2 - affine - softmax**
![HSI_net](https://github.com/ShoupingShan/Data_fusion_HSI_LiDAR/blob/master/image/HSI_net.png?raw=true)
![LiDAR](https://github.com/ShoupingShan/Data_fusion_HSI_LiDAR/blob/master/image/LiDAr-DSM_net.png?raw=true)
## Files
  `./HSI/Load_data.py  ` Load HSI source data and make Train/Test files as patch

  `./HSI/CNN.py ` define CNN parameters
  `./HSI/CNN_feed.py`  Train HSI CNN weights
  `./HSI/run_cnn.py`  HSI classification using pre-trained CNN parameters
  `./HSI/Spatial_dataset.py `  Provides a highly flexible Dataset class for handling the HSI data.
  `./HSI/Get_feature.py`  Save last-pooling-layer-flat feature
  `./DSM`  Almost the same as `./HSI`

## Result

![GT](https://github.com/ShoupingShan/Data_fusion_HSI_LiDAR/blob/master/HSI/result/Gt.png?raw=true)
![HSI_result](https://github.com/ShoupingShan/Data_fusion_HSI_LiDAR/blob/master/HSI/result/Map.png?raw=true)
## Contact
[shp395210@outlook.com](shp395210@outlook.com)
