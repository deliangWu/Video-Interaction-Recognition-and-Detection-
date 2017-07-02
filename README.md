# Video-Interaction-Recognition-and-Detection-
This is a master thesis project aiming for interaction classification and detection in videos. So, there are two main parts in the projects, including interaction classification and interaction detection.   

Software dependencies: 
----------------------------------------
1. Python 2 or Python 3  
2. TensorFlow r1.2  
3. opencv 3.2.0  
4. numpy  
5. xlrd  
6. matplotlib.pyplot  
7. imutils  


The file strcture:
----------------------------------------
 * Project  
  * - datasets:   The UT-Interaction dataset (videos)  
  * - log:        The logging files which have all training and evaluating records  
  * - variables:  The viarible parameter files  
  * -src/  
   * - common:     Common files   
   * - datasets:   The python files for data pre-processing  
   * - dataVisualization: 3D CNN data visualization, and evaluating results visualization
   * - model:      3D ConvNet basic model components.   
   * - train:      Interaction classification networks, including singleNet, fullNet, pre-trained Net, etc.  
   * - humanSegmentation: person detection, tracking and segmentation  
   * - detector:   interaction detection.   
        

