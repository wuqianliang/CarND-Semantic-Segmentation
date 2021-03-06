# Semantic Segmentation Project
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

    def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
    
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

in function layers, we first add 1x1 conv by pool7 and upsample to get layer7_up, add skip of pool4 1x1 conv and layer7_up,then do the same to pool4, then do upsample to layer3_skip to get final modified model. I do the same way as FCN paper described.

#### Does the project optimize the neural network?
I use the cross_entropy_loss and L2 regularization penalty on loss.

    def optimize(nn_last_layer, correct_label, learning_rate, num_classes, l2_const)

        logits = tf.reshape(nn_last_layer, [-1,num_classes])
        labels = tf.reshape(correct_label,[-1,num_classes])

        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam')

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = cross_entropy_loss + l2_const * sum(reg_losses)
        train_op = optimizer.minimize(loss=loss)

        return logits, train_op, loss
    
#### Does the project train the neural network?
Yes during trainning, the model script print loss:

    Starting run for kp_5E-01,lr_1E-04,l2_1E-02
    2018-06-23 16:47:07.441711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0)
    loss 0.38728118  epoch_i  1  epochs  200  counter  20
    loss 0.18524675  epoch_i  2  epochs  200  counter  40
    loss 0.16316253  epoch_i  3  epochs  200  counter  60
    loss 0.14614113  epoch_i  4  epochs  200  counter  80
    loss 0.16411816  epoch_i  5  epochs  200  counter  100
    loss 0.13614361  epoch_i  6  epochs  200  counter  120
    loss 0.11559493  epoch_i  7  epochs  200  counter  140
    loss 0.11490484  epoch_i  8  epochs  200  counter  160
    loss 0.0668481  epoch_i  9  epochs  200  counter  180
    loss 0.06830554  epoch_i  10  epochs  200  counter  200
    loss 0.06855286  epoch_i  11  epochs  200  counter  220
    loss 0.06652438  epoch_i  12  epochs  200  counter  240
    loss 0.07009909  epoch_i  13  epochs  200  counter  260
    loss 0.045881167  epoch_i  14  epochs  200  counter  280
    loss 0.057495188  epoch_i  15  epochs  200  counter  300
    loss 0.07169177  epoch_i  16  epochs  200  counter  320
    loss 0.06764566  epoch_i  17  epochs  200  counter  340
    loss 0.05766602  epoch_i  18  epochs  200  counter  360
    loss 0.038915638  epoch_i  19  epochs  200  counter  380
    loss 0.03408567  epoch_i  21  epochs  200  counter  400
    loss 0.05065194  epoch_i  22  epochs  200  counter  420
    loss 0.041606896  epoch_i  23  epochs  200  counter  440


#### Does the project train the model correctly?
Yes, the loss decreases when trainning:

    loss 0.017175457 epoch_i 188 epochs 200 counter 3580 
    loss 0.017048566 epoch_i 189 epochs 200 counter 3600 
    loss 0.01274084 epoch_i 190 epochs 200 counter 3620 
    loss 0.015761087 epoch_i 191 epochs 200 counter 3640 
    loss 0.009665441 epoch_i 192 epochs 200 counter 3660 
    loss 0.009468366 epoch_i 193 epochs 200 counter 3680 
    loss 0.009596125 epoch_i 194 epochs 200 counter 3700 
    loss 0.009907771 epoch_i 195 epochs 200 counter 3720 
    loss 0.00977212 epoch_i 196 epochs 200 counter 3740 
    loss 0.010752921 epoch_i 197 epochs 200 counter 3760 
    loss 0.008895747 epoch_i 198 epochs 200 counter 3780 
    loss 0.005522505 epoch_i 199 epochs 200 counter 3800 
    Model saved in path: ./models/kp_5E-01,lr_1E-04,l2_5E-03/model
with keep_portion=0.5,learning_rate=0.0001 and l2_coeff=0.005, model get the lowest train loss.
    
#### Does the project use reasonable hyperparameters?
Yes, I do the hyperparameters searching of the learning rate ,keep proportion,and the const coeff of L2-Loss.
As we see after searching, we found that 
    learning rate= 0.0001
    keep proportion = 0.5
    L2-Loss coeff = 0.01
made final loss=0.0058767293 after 200 round trainning.

#### Does the project correctly label the road?
Yes,the following labeled images showed the label effects.

![Alt text](https://github.com/wuqianliang/CarND-Semantic-Segmentation/blob/master/images/um_000044.png "Optional title")
![Alt text](https://github.com/wuqianliang/CarND-Semantic-Segmentation/blob/master/images/uu_000024.png "Optional title")
![Alt text](https://github.com/wuqianliang/CarND-Semantic-Segmentation/blob/master/images/uu_000092.png "Optional title")


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

### Reference 
https://github.com/ksakmann/CarND-Semantic-Segmentation
