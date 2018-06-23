# Semantic Segmentation Porject
Self-Driving Car Engineer Nanodegree Program

---
### Overview
In this project ,I implemented a model to make pixel level classification.I use 1x1 convolution on the vgg_layer7,vgg_layer4,vgg_layer3 ,upsample these three layers and iteratively build skip layers between upsampled and original pool layers. Also I used normalized initial parameters and regularization l2 losses in every new layer.

### Project steps
1. Pepraring the vgg pretrained model and  trainning & testing datasets.
2. Load pretrained model and use 1x1 conv,upsample and skip layers to build new network.
3. Build tf objects like loss, loss minimization etc.
4. Use batch_size=16 and epochs=200 to train modified layer, and also do hyperparameter(learning rate,keep proportion,l2 coeff) search.
5. Make inference on test images.

### Rubic
#### Does the project load the pretrained vgg model?
Yes, in `line 21` and `line 240` we implement and invoke the `load_vgg()` function to load pretrained vgg model and get the pool_7,pool_4,pool_3 layers and input layer.  
#### Does the project learn the correct features from the images?
Yes, I implement final layer like following:
`
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    layer7_1x1 = conv_1x1(vgg_layer7_out,num_classes)
    layer4_1x1 = conv_1x1(vgg_layer4_out,num_classes)
    layer3_1x1 = conv_1x1(vgg_layer3_out,num_classes)

    # FCN-32s
    layer7_up = upsample(layer7_1x1,num_classes,5,2) 
    layer4_skip = skip_layer(layer7_up,layer4_1x1)

    # FCN-16s
    layer4_up = upsample(layer4_skip,num_classes,5,2)
    layer3_skip = skip_layer(layer4_up,layer3_1x1)

    # # FCN-8s
    model = upsample(layer3_skip, num_classes,16,8)
    
    return model
`
#### Does the project optimize the neural network?

#### Does the project train the neural network?

#### Does the project train the model correctly?

#### Does the project use reasonable hyperparameters?

#### Does the project correctly label the road?


---
# Original Readme
# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
