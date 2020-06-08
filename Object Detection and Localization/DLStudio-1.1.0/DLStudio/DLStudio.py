# -*- coding: utf-8 -*-

__version__   = '1.1.0'
__author__    = "Avinash Kak (kak@purdue.edu)"
__date__      = '2020-March-28'   
__url__       = 'https://engineering.purdue.edu/kak/distDLS/DLStudio-1.1.0.html'
__copyright__ = "(C) 2020 Avinash Kak. Python Software Foundation."

__doc__ = '''

DLStudio.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@title
CHANGE LOG:

  Version 1.1.0:

    The main reason for this version was my observation that when the
    training data is intentionally corrupted with a high level of noise, it
    is possible for the output of regression to be a NaN (Not a Number).
    In my testing at noise levels of 20%, 50%, and 80%, while you do not
    see this problem when the noise level is 20%, it definitely becomes a
    problem when the noise level is at 50%.  To deal with this issue, this
    version includes the test 'torch.isnan()' in the training and testing
    code for object detection.  This version of the module also provides
    additional datasets with noise corrupted images with different levels
    of noise.  However, since the total size of the datasets now exceeds
    the file-size limit at 'https://pypi.org', you'll need to download them
    separately from the link provided in the main documentation page.

  Version 1.0.9:

    With this version, you can now use DLStudio for experiments in semantic
    segmentation of images.  The code added to the module is in a new inner
    class that, as you might guess, is named SemanticSegmentation.  The
    workhorse of this inner class is a new implementation of the famous
    Unet that I have named mUnet --- the prefix "m" stands for "multi" for
    the ability of the network to segment out multiple objects
    simultaneously.  This version of DLStudio also comes with a new
    dataset, PurdueShapes5MultiObject, for experimenting with mUnet.  Each
    image in this dataset contains a random number of selections from five
    different shapes --- rectangle, triangle, disk, oval, and star --- that
    are randomly scaled, oriented, and located in each image.

  Version 1.0.7:

    The main reason for creating this version of DLStudio is to be able to
    use the module for illustrating how to simultaneously carry out
    classification and regression (C&R) with the same convolutional
    network.  The specific C&R problem that is solved in this version is
    the problem of object detection and localization. You want a CNN to
    categorize the object in an image and, at the same time, estimate the
    bounding-box for the detected object. Estimating the bounding-box is
    referred to as regression.  All of the code related to object detection
    and localization is in the inner class DetectAndLocalize of the main
    module file.  Training a CNN to solve the detection and localization
    problem requires a dataset that, in addition to the class labels for
    the objects, also provides bounding-box annotations for the objects.
    Towards that end, this version also comes with a new dataset called
    PurdueShapes5.  Another new inner class, CustomDataLoading, that is
    also included in Version 1.0.7 has the dataloader for the PurdueShapes5
    dataset.

  Version 1.0.6:

    This version has the bugfix for a bug in SkipBlock that was spotted by
    a student as I was demonstrating in class the concepts related to the
    use of skip connections in deep neural networks.

  Version 1.0.5:

    This version includes an inner class, SkipConnections, for
    experimenting with skip connections to improve the performance of a
    deep network.  The Examples subdirectory of the distribution includes a
    script, playing_with_skip_connections.py, that demonstrates how you can
    experiment with SkipConnections.  The network class used by
    SkipConnections is named BMEnet with an easy-to-use interface for
    experimenting with networks of arbitrary depth.

  Version 1.0.4:

    I have added one more inner class, AutogradCustomization, to the module
    that illustrates how to extend Autograd if you want to endow it with
    additional functionality. And, most importantly, this version fixes an
    important bug that caused wrong information to be written out to the
    disk when you tried to save the learned model at the end of a training
    session. I have also cleaned up the comment blocks in the
    implementation code.

  Version 1.0.3:

    This is the first public release version of this module.

@title
INTRODUCTION:

    Every design activity involves mixing and matching things and doing so
    repeatedly until you have achieved the desired results.  The same thing
    is true of modern deep learning networks.  When you are working with a
    new data domain, it is likely that you would want to experiment with
    different network layouts that you may have dreamed of yourself or that
    you may have seen somewhere in a publication or at some web site.

    The goal of this module is to make it easier to engage in this process.
    The idea is that you would drop in the module a new network and you
    would be able to see right away the results you would get with the new
    network.

    This module also allows you to specify a network with a configuration
    string.  The module parses the string and creates the network.  In
    upcoming revisions of this module, I am planning to add additional
    features to this approach in order to make it more general and more
    useful for production work.


    EXTENDING AUTOGRAD:

    Version 1.0.4 of DLStudio incorporates a new inner class,
    AutogradCustomization, for illustrating how you can write your own code
    for customizing the behavior of PyTorch's Autograd module. Your
    starting point for understanding the code in AutogradCustomization
    should be the following script in the Examples directory of the distro:

                extending_autograd.py

    Extending Autograd requires that you define a new verb class --- as I
    have with the class DoSillyWithTensor shown in the main module file ---
    with definitions for two static methods, "forward()" and "backward()".
    Note that an instance constructed from this class is callable.


    SKIP CONNECTIONS:

    Starting with Version 1.0.6, you can now experiment with skip
    connections in a CNN to see how a deep network with this feature might
    yield improved classification results.  Deep networks suffer from the
    problem of vanishing gradients that degrades their performance.
    Vanishing gradients means that the gradients of the loss calculated in
    the early layers of a network become increasingly muted as the network
    becomes deeper.  An important mitigation strategy for addressing this
    problem consists of creating a CNN using blocks with skip connections.

    The code for using skip connections is in the inner class
    SkipConnections of the module.  And the network that allows you to
    construct a CNN with skip connections is named BMEnet.  As shown in the
    script playing_with_skip_connections.py in the Examples directory of
    the distribution, you can easily create a CNN with arbitrary depth just
    by using the constructor option "depth" for BMEnet. The basic block of
    the network constructed in this manner is called SkipBlock which, very
    much like the BasicBlock in ResNet-18, has a couple of convolutional
    layers whose output is combined with the input to the block.

    Note that the value given to the "depth" constructor option for the
    BMEnet class does NOT translate directly into the actual depth of the
    CNN. [Again, see the script playing_with_skip_connections.py in the
    Examples directory for how to use this option.] The value of "depth" is
    translated into how many instances of SkipBlock to use for constructing
    the CNN.

    If you want to use DLStudio for learning how to create your own
    versions of SkipBlock-like shortcuts in a CNN, your starting point
    should be the following script in the Examples directory of the distro:

                playing_with_skip_connections.py

    This script illustrates how to use the inner class BMEnet of the module
    for experimenting with skip connections in a CNN. As the script shows,
    the constructor of the BMEnet class comes with two options:
    skip_connections and depth.  By turning the first on and off, you can
    directly illustrate in a classroom setting the improvement you can get
    with skip connections.  And by giving an appropriate value to the
    "depth" option, you can show results for networks of different depths.


    OBJECT DETECTION AND LOCALIZATION:

    The code for how to solve the problem of object detection and
    localization with a CNN is in the inner classes DetectAndLocalize and
    CustomDataLoading.  This code was developed for version 1.0.7 of the
    module.  In general, object detection and localization problems are
    more challenging than pure classification problems because solving the
    localization part requires regression for the coordinates of the
    bounding box that localize the object.  If at all possible, you would
    want the same CNN to provide answers to both the classification and the
    regression questions and do so at the same time.  This calls for a CNN
    to possess two different output layers, one for classification and the
    other for regression.  A deep network that does exactly that is
    illustrated by the LOADnet classes that are defined in the inner class
    DetectAndLocalize of the DLStudio module.  [By the way, the acronym
    "LOAD" in "LOADnet" stands for "LOcalization And Detection".] Although
    you will find three versions of the LOADnet class inside
    DetectAndLocalize, for now only pay attention to the LOADnet2 class
    since that is the one I have worked with the most for creating the
    1.0.7 distribution.

    As you would expect, training a CNN for object detection and
    localization requires a dataset that, in addition to the class labels
    for the images, also provides bounding-box annotations for the objects
    in the images. Out of my great admiration for the CIFAR-10 dataset as
    an educational tool for solving classification problems, I have created
    small-image-format training and testing datasets for illustrating the
    code devoted to object detection and localization in this module.  The
    training dataset is named PurdueShapes5-10000-train.gz and it consists
    of 10,000 images, with each image of size 32x32 containing one of five
    possible shapes --- rectangle, triangle, disk, oval, and star. The
    shape objects in the images are randomized with respect to size,
    orientation, and color.  The testing dataset is named
    PurdueShapes5-1000-test.gz and it contains 1000 images generated by the
    same randomization process as used for the training dataset.  You will
    find these datasets in the "data" subdirectory of the "Examples"
    directory in the distribution.

    Providing a new dataset for experiments with detection and localization
    meant that I also needed to supply a custom dataloader for the dataset.
    Toward that end, Version 1.0.7 also includes another inner class named
    CustomDataLoading where you will my implementation of the custom
    dataloader for the PurdueShapes5 dataset.

    If you want to use DLStudio for learning how to write your own PyTorch
    code for object detection and localization, your starting point should
    be the following script in the Examples directory of the distro:

                object_detection_and_localization.py

    Execute the script and understand what functionality of the inner class
    DetectAndLocalize it invokes for object detection and localization.


    OBJECT DETECTION AND LOCALIZATION IN THE
    PRESENCE OF SIGNIFICANT NOISE:

    When the training data is intentionally corrupted with a high level of
    noise, it is possible for the output of regression to be a NaN (Not a
    Number).  Here is what I observed when I tested the LOADnet2 network at
    noise levels of 20%, 50%, and 80%: At 20% noise, both the labeling and
    the regression accuracies become worse compared to the noiseless case,
    but they would still be usable depending on the application.  For
    example, with two epochs of training, the overall classification
    accuracy decreases from 91% to 83% and the regression error increases
    from under a pixel (on the average) to around 3 pixels.  However, when
    the level of noise is increased to 50%, the regression output is often
    a NaN (Not a Number), as presented by 'numpy.nan' or 'torch.nan'.  To
    deal with this problem, Version 1.1.0 of the DLStudio module checks the
    output of the bounding-box regression before drawing the rectangles on
    the images.  

    If you wish to experiment with detection and localization in the
    presence of noise, your starting point should be the script

                noisy_object_detection_and_localization.py

    in the Examples directory of the distribution.  Note that you would
    need to download the datasets for such experiments directly from the
    link provided near the top of this documentation page.


    SEMANTIC SEGMENTATION:

    The code for how to carry out semantic segmentation is in the inner
    class that is appropriately named SemanticSegmentation.  At its
    simplest, the purpose of semantic segmentation is to assign correct
    labels to the different objects in a scene, while localizing them at
    the same time.  At a more sophisticated level, a system that carries
    out semantic segmentation should also output a symbolic expression that
    reflects an understanding of the scene in the image that is based on
    the objects found in the image and their spatial relationships with one
    another.  The code in the new inner class is based on only the simplest
    possible definition of what is meant by semantic segmentation.
    
    The convolutional network that carries out semantic segmentation
    DLStudio is named mUnet, where the letter "m" is short for "multi",
    which, in turn, stands for the fact that mUnet is capable of segmenting
    out multiple object simultaneously from an image.  The mUnet network is
    based on the now famous Unet network that was first proposed by
    Ronneberger, Fischer and Brox in the paper "U-Net: Convolutional
    Networks for Biomedical Image Segmentation".  Their UNET extracts
    binary masks for the cell pixel blobs of interest in biomedical images.
    The output of UNET can therefore be treated as a pixel-wise binary
    classifier at each pixel position.  The mUnet class, on the other hand,
    is intended for segmenting out multiple objects simultaneously form an
    image. [A weaker reason for "m" in the name of the class is that it
    uses skip connections in multiple ways --- such connections are used
    not only across the two arms of the "U", but also also along the arms.
    The skip connections in the original Unet are only between the two arms
    of the U.  

    mUnet works by assigning a separate channel in the output of the
    network to each different object type.  After the network is trained,
    for a given input image, all you have to do is examine the different
    channels of the output for the presence or the absence of the objects
    corresponding to the channel index.

    This version of DLStudio also comes with a new dataset,
    PurdueShapes5MultiObject, for experimenting with mUnet.  Each image
    in this dataset contains a random number of selections from five
    different shapes, with the shapes being randomly scaled, oriented, and
    located in each image.  The five different shapes are: rectangle,
    triangle, disk, oval, and star.

    Your starting point for learning how to use the mUnet network for
    segmenting images should be the following script in the Examples
    directory of the distro:

                semantic_segmentation.py

    Execute the script and understand how it uses the functionality packed
    in the inner class SemanticSegmentation for segmenting out the objects
    in an image.


@title
INSTALLATION:

    The DLStudio class was packaged using setuptools.  For
    installation, execute the following command in the source directory
    (this is the directory that contains the setup.py file after you have
    downloaded and uncompressed the package):
 
            sudo python setup.py install

    and/or, for the case of Python3, 

            sudo python3 setup.py install

    On Linux distributions, this will install the module file at a location
    that looks like

             /usr/local/lib/python2.7/dist-packages/

    and, for the case of Python3, at a location that looks like

             /usr/local/lib/python3.6/dist-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the DLStudio class:

            import sys
            sys.path.append( "pathname_to_DLStudio_directory" )

    To uninstall the module, simply delete the source directory, locate
    where the DLStudio module was installed with "locate
    DLStudio" and delete those files.  As mentioned above,
    the full pathname to the installed version is likely to look like
    /usr/local/lib/python2.7/dist-packages/DLStudio*

    If you want to carry out a non-standard install of the
    DLStudio module, look up the on-line information on
    Disutils by pointing your browser to

              http://docs.python.org/dist/dist.html

@title
USAGE:

    If you want to specify a network with just a configuration string,
    your usage of the module is going to look like:


        from DLStudio import *
        
        convo_layers_config = "1x[128,3,3,1]-MaxPool(2) 1x[16,5,5,1]-MaxPool(2)"
        fc_layers_config = [-1,1024,10]
        
        dls = DLStudio(   dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                          image_size = [32,32],
                          convo_layers_config = convo_layers_config,
                          fc_layers_config = fc_layers_config,
                          path_saved_model = "./saved_model",
                          momentum = 0.9,
                          learning_rate = 1e-3,
                          epochs = 2,
                          batch_size = 4,
                          classes = ('plane','car','bird','cat','deer',
                                     'dog','frog','horse','ship','truck'),
                          use_gpu = True,
                          debug_train = 0,
                          debug_test = 1,
                      )
        
        configs_for_all_convo_layers = dls.parse_config_string_for_convo_layers()
        convo_layers = dls.build_convo_layers2( configs_for_all_convo_layers )
        fc_layers = dls.build_fc_layers()
        model = dls.Net(convo_layers, fc_layers)
        dls.show_network_summary(model)
        dls.load_cifar_10_dataset()
        dls.run_code_for_training(model)
        dls.run_code_for_testing(model)
                

    or, if you would rather experiment with a drop-in network, your usage
    of the module is going to look something like:


        dls = DLStudio(   dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                          image_size = [32,32],
                          path_saved_model = "./saved_model",
                          momentum = 0.9,
                          learning_rate = 1e-3,
                          epochs = 2,
                          batch_size = 4,
                          classes = ('plane','car','bird','cat','deer',
                                     'dog','frog','horse','ship','truck'),
                          use_gpu = True,
                          debug_train = 0,
                          debug_test = 1,
                      )
        
        exp_seq = DLStudio.ExperimentsWithSequential( dl_studio = dls )   ## for your drop-in network
        exp_seq.load_cifar_10_dataset_with_augmentation()
        model = exp_seq.Net()
        dls.show_network_summary(model)
        exp_seq.run_code_for_training(model)
        exp_seq.run_code_for_testing(model)

        
    This assumes that you copy-and-pasted the network you want to
    experiment with in a class like ExperimentsWithSequential that is
    included in the module.


@title
CONSTRUCTOR PARAMETERS: 

    batch_size:  Carries the usual meaning in the neural network context.

    classes:  A list of the symbolic names for the classes.

    convo_layers_config: This parameter allows you to specify a convolutional network
                  with a configuration string.  Must be formatted as explained in the
                  comment block associated with the method
                  "parse_config_string_for_convo_layers()"

    dataroot: This points to where your dataset is located.

    debug_test: Setting it allow you to see images being used and their predicted
                 class labels every 2000 batch-based iterations of testing.

    debug_train: Does the same thing during training that debug_test does during
                 testing.

    epochs: Specifies the number of epochs to be used for training the network.

    fc_layers_config: This parameter allows you to specify the final
                 fully-connected portion of the network with just a list of
                 the number of nodes in each layer of this portion.  The
                 first entry in this list must be the number '-1', which
                 stands for the fact that the number of nodes in the first
                 layer will be determined by the final activation volume of
                 the convolutional portion of the network.

    image_size:  The heightxwidth size of the images in your dataset.

    learning_rate:  Again carries the usual meaning.

    momentum:  Carries the usual meaning and needed by the optimizer.

    path_saved_model: The path to where you want the trained model to be
                  saved in your disk so that it can be retrieved later
                  for inference.

    use_gpu: You must set it to True if you want the GPU to be used for training.


@title
PUBLIC METHODS:

    (1)  build_convo_layers()

         This method creates the convolutional layers from the parameters
         in the configuration string that was supplied through the
         constructor option 'convo_layers_config'.  The output produced by
         the call to 'parse_config_string_for_convo_layers()' is supplied
         as the argument to build_convo_layers().

    (2)  build_fc_layers()

         From the list of ints supplied through the constructor option
         'fc_layers_config', this method constructs the fully-connected
         portion of the overall network.

    (3)  check_a_sampling_of_images()        

         Displays the first batch_size number of images in your dataset.


    (4)  display_tensor_as_image()

         This method will display any tensor of shape (3,H,W), (1,H,W), or
         just (H,W) as an image. If any further data normalizations is
         needed for constructing a displayable image, the method takes care
         of that.  It has two input parameters: one for the tensor you want
         displayed as an image and the other for a title for the image
         display.  The latter parameter is default initialized to an empty
         string.

    (5)  load_cifar_10_dataset()

         This is just a convenience method that calls on Torchvision's
         functionality for creating a data loader.

    (6)  load_cifar_10_dataset_with_augmentation()             

         This convenience method also creates a data loader but it also
         includes the syntax for data augmentation.

    (7)  parse_config_string_for_convo_layers()

         As mentioned in the Introduction, DLStudio module allows you to
         specify a convolutional network with a string provided the string
         obeys the formatting convention described in the comment block of
         this method.  This method is for parsing such a string. The string
         itself is presented to the module through the constructor option
         'convo_layers_config'.

    (8)  run_code_for_testing()

         This is the method runs the trained model on the test data. Its
         output is a confusion matrix for the classes and the overall
         accuracy for each class.  The method has one input parameter which
         is set to the network to be tested.  This learnable parameters in
         the network are initialized with the disk-stored version of the
         trained model.

    (9)  run_code_for_training()

         This is the method that does all the training work. If a GPU was
         detected at the time an instance of the module was created, this
         method takes care of making the appropriate calls in order to
         transfer the tensors involved into the GPU memory.

    (10) save_model()

         Writes the model out to the disk at the location specified by the
         constructor option 'path_saved_model'.  Has one input parameter
         for the model that needs to be written out.

    (11) show_network_summary()

         Displays a print representation of your network and calls on the
         torchsummary module to print out the shape of the tensor at the
         output of each layer in the network. The method has one input
         parameter which is set to the network whose summary you want to
         see.


@title 
INNER CLASSES OF THE MODULE:

    The purpose of the following two inner classes is to demonstrate how
    you can create a custom class for your own network and test it within
    the framework provided by the DLStudio module.

    (1)  class ExperimentsWithSequential

         This class is my demonstration of experimenting with a network
         that I found on GitHub.  I copy-and-pasted it in this class to
         test its capabilities.  How to call on such a custom class is
         shown by the following script in the Examples directory:

                     playing_with_sequential.py

    (2)  class ExperimentsWithCIFAR

         This is very similar to the previous inner class, but uses a
         common example of a network for experimenting with the CIFAR-10
         dataset. Consisting of 32x32 images, this is a great dataset for
         creating classroom demonstrations of convolutional networks.
         As to how you should use this class is shown in the following
         script

                    playing_with_cifar10.py

         in the Examples directory of the distribution.

    (3)  class AutogradCustomization

         The purpose of this class is to illustrate how to extend Autograd
         with additional functionality. What's shown is an implementation of 
         the recommended approach at the following documentation page:

               https://pytorch.org/docs/stable/notes/extending.html

    (4)  class SkipConnections

         This class is for investigating the power of skip connections in
         deep networks.  Skip connections are used to mitigate a serious
         problem associated with deep networks --- the problem of vanishing
         gradients.  It has been argued theoretically and demonstrated
         empirically that as the depth of a neural network increases, the
         gradients of the loss become more and more muted for the early
         layers in the network.

    (5)  class DetectAndLocalize

         The code in this inner class is for demonstrating how the same
         convolutional network can simultaneously the twin problems of
         object detection and localization.  Note that, unlike the previous
         four inner classes, class DetectAndLocalize comes with its own
         implementations for the training and testing methods. The main
         reason for that is that the training for detection and
         localization must use two different loss functions simultaneously,
         one for classification of the objects and the other for
         regression. The function for testing is also a bit more involved
         since it must now compute two kinds of errors, the classification
         error and the regression error on the unseen data. Although you
         will find a couple of different choices for the training and
         testing functions for detection and localization inside
         DetectAndLocalize, the ones I have worked with the most are those
         that are used in the following two scripts in the Examples
         directory:

              run_code_for_training_with_CrossEntropy_and_MSE_Losses()

              run_code_for_testing_detection_and_localization()

    (6)  class CustomDataLoading

         This is a testbed for experimenting with a completely grounds-up
         attempt at designing a custom data loader.  Ordinarily, if the
         basic format of how the dataset is stored is similar to one of the
         datasets that Torchvision knows about, you can go ahead and use
         that for your own dataset.  At worst, you may need to carry out
         some light customizations depending on the number of classes
         involved, etc.  However, if the underlying dataset is stored in a
         manner that does not look like anything in Torchvision, you have
         no choice but to supply yourself all of the data loading
         infrastructure.  That is what this inner class of the DLStudio
         module is all about.

    (7)  class SemanticSegmentation

         This inner class is for working with the mUnet convolutional
         network for semantic segmentation of images.  This network allows
         you to segment out multiple objects simultaneously from an image.
         Each object type is assigned a different channel in the output of
         the network.  So, for segmenting out the objects of a specified
         type in a given input image, all you have to do is examine the
         corresponding channel in the output.


@title 
THE Examples DIRECTORY:

    The Examples subdirectory in the distribution contains the following
    three scripts:

    (1)  playing_with_reconfig.py

         Shows how you can specify a convolution network with a
         configuration string.  The DLStudio module parses the string
         constructs the network.

    (2)  playing_with_sequential.py

         Shows you how you can call on a custom inner class of the
         'DLStudio' module that is meant to experiment with your own
         network.  The name of the inner class in this example script is
         ExperimentsWithSequential

    (3)  playing_with_cifar10.py

         This is very similar to the previous example script but is based
         on the inner class ExperimentsWithCIFAR which uses more common
         examples of networks for playing with the CIFAR-10 dataset.

    (4)  extending_autograd.py

         This provides a demonstration example of the recommended approach
         for giving additional functionality to Autograd --- as mentioned
         in the commented made above about the inner class
         AutogradCustomization.

    (5)  playing_with_skip_connections.py

         This script illustrates how to use the inner class BMEnet of the
         module for experimenting with skip connections in a CNN. As the
         script shows, the constructor of the BMEnet class comes with two
         options: skip_connections and depth.  By turning the first on and
         off, you can directly illustrate in a classroom setting the
         improvement you can get with skip connections.  And by giving an
         appropriate value to the "depth" option, you can show results for
         networks of different depths.

    (6)  custom_data_loading.py

         This script shows how to use the custom dataloader in the inner
         class CustomDataLoading of the DLStudio module.  That custom
         dataloader is meant specifically for the PurdueShapes5 dataset
         that is used in object detection and localization experiments in
         DLStudio.

    (7)  object_detection_and_localization.py

         This script shows how you can use the functionality provided by
         the inner class DetectAndLocalize of the DLStudio module for
         experimenting with object detection and localization.  Detecting
         and localizing (D&L) objects in images is a more difficult problem
         than just classifying the objects.  D&L requires that your CNN
         make two different types of inferences simultaneously, one for
         classification and the other for localization.  For the
         localization part, the CNN must carry out what is known as
         regression. What that means is that the CNN must output the
         numerical values for the bounding box that encloses the object
         that was detected.  Generating these two types of inferences
         requires two different loss functions, one for classification and
         the other for regression.

    (8)  noisy_object_detection_and_localization.py

         This script in the Examples directory is exactly the same as the
         one described above, the only difference is that it calls on the
         noise-corrupted training and testing dataset files.  I thought it
         would be best to create a separate script for studying the effects
         of noise, just to allow for the possibility that the noise-related
         studies with DLStudio may evolve differently in the future.

    (9)  semantic_segmentation.py

         This script should be your starting point if you wish to learn how
         to use the mUnet neural network for semantic segmentation of
         images.  As mentioned elsewhere in this documentation page, mUnet
         assigns an output channel to each different type of object that
         you wish to segment out from an image. So, given a test image at
         the input to the network, all you have to do is to examine each
         channel at the output for segmenting out the objects that
         correspond to that output channel.


@title 
THE DATASETS INCLUDED (must be downloaded separately):

    Download the dataset archive 'datasets_for_DLStudio.tar.gz' from the
    link provided at the top of this documentation page and store it in the
    'Example' directory of the distribution.  Subsequently, execute the
    following command in the 'Examples' directory:

        cd Examples
        tar xvf datasets_for_DLStudio.tar.gz

    This command will create a 'data' subdirectory of 'Examples' and
    deposit in the subdirectory the datasets mentioned below.


    OBJECT DETECTION AND LOCALIZATION:

    Training a CNN for object detection and localization requires training
    and testing datasets that come with bounding-box annotations. This
    module comes with the PurdueShapes5 dataset for that purpose.  I
    created this small-image-format dataset out of my admiration for the
    CIFAR-10 dataset as an educational tool for demonstrating
    classification networks in a classroom setting. You will find the
    following dataset archive files in the "data" subdirectory of the
    "Examples" directory of the distro:

        (1)  PurdueShapes5-10000-train.gz
        (2)  PurdueShapes5-1000-test.gz
        (3)  PurdueShapes5-20-train.gz
        (4)  PurdueShapes5-20-test.gz               

    The number that follows the main name string "PurdueShapes5-" is for
    the number of images in the dataset.  You will find the last two
    datasets, with 20 images each, useful for debugging your logic for
    object detection and bounding-box regression.

    As to how the image data is stored in the archives, please see the main
    comment block for the inner class CustomLoading in this file.


    OBJECT DETECTION AND LOCALIZATION
    WITH NOISE-CORRUPTED IMAGES:

    In terms of how the image data is stored in the dataset files, this
    dataset is no different from the PurdueShapes5 dataset described above.
    The only difference is that we now add varying degrees of noise to the
    images to make it more challenging for both classification and
    regression.

    The archive files you will find in the 'data' subdirectory of the
    'Examples' directory for this dataset are:

        (5)  PurdueShapes5-10000-train-noise-20.gz
        (6)  PurdueShapes5-10000-train-noise-50.gz
        (7)  PurdueShapes5-10000-train-noise-80.gz
        (8)  PurdueShapes5-1000-test-noise-20.gz
        (9)  PurdueShapes5-1000-test-noise-50.gz
        (10) PurdueShapes5-1000-test-noise-80.gz

    In the names of these six archive files, the numbers 20, 50, and 80
    stand for the level of noise in the images.  For example, 20 means 20%
    noise.  The percentage level indicates the fraction of the color value
    range that is added as randomly generated noise to the images.  The
    first integer in the name of each archive carries the same meaning as
    mentioned above for the regular PurdueShapes5 dataset: It stands for
    the number of images in the dataset.


    SEMANTIC SEGMENTATION:

    Showing interesting results with semantic segmentation requires images
    that contains multiple objects of different types.  A good semantic
    segmenter would then allow for each object type to be segmented out
    separately from an image.  A network that can carry out such
    segmentation needs training and testing datasets in which the images
    come up with multiple objects of different types in them. Towards that
    end, I have created the following dataset:

        (11) PurdueShapes5MultiObject-10000-train.gz
        (12) PurdueShapes5MultiObject-1000-test.gz
        (13) PurdueShapes5MultiObject-20-train.gz
        (14) PurdueShapes5MultiObject-20-test.gz

    The number that follows the main name string
    "PurdueShapes5MultiObject-" is for the number of images in the dataset.
    You will find the last two datasets, with 20 images each, useful for
    debugging your logic for semantic segmentation.

    As to how the image data is stored in the archive files listed above,
    please see the main comment block for the class

        PurdueShapes5MultiObjectDataset

    As explained there, in addition to the RGB values at the pixels that
    are stored in the form of three separate lists called R, G, and B, the
    shapes themselves are stored in the form an array of masks, each of
    size 64x64, with each mask array representing a particular shape. For
    illustration, the rectangle shape is represented by the first such
    array. And so on.


@title 
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'DLStudio' in the subject line to get
    past the author's spam filter.


@title
ABOUT THE AUTHOR:

    The author, Avinash Kak, is a professor of Electrical and Computer
    Engineering at Purdue University.  For all issues related to this
    module, contact the author at kak@purdue.edu If you send email, please
    place the string "DLStudio" in your subject line to get past the
    author's spam filter.


@title
COPYRIGHT:

    Python Software Foundation License

    Copyright 2020 Avinash Kak

@endofdocs
'''


import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import pymsgbox

#______________________________  DLStudio Class Definition  ________________________________

class DLStudio(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''DLStudio constructor can only be called with keyword arguments for 
                      the following keywords: epochs, learning_rate, batch_size, momentum,
                      convo_layers_config, image_size, dataroot, path_saved_model, classes, 
                      image_size, convo_layers_config, fc_layers_config, debug_train, use_gpu, and 
                      debug_test''')
        learning_rate = epochs = batch_size = convo_layers_config = momentum = None
        image_size = fc_layers_config = dataroot =  path_saved_model = classes = use_gpu = None
        debug_train  = debug_test = None
        if 'dataroot' in kwargs                      :   dataroot = kwargs.pop('dataroot')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'convo_layers_config' in kwargs           :   convo_layers_config = kwargs.pop('convo_layers_config')
        if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
        if 'fc_layers_config' in kwargs              :   fc_layers_config = kwargs.pop('fc_layers_config')
        if 'path_saved_model' in kwargs              :   path_saved_model = kwargs.pop('path_saved_model')
        if 'classes' in kwargs                       :   classes = kwargs.pop('classes') 
        if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu') 
        if 'debug_train' in kwargs                   :   debug_train = kwargs.pop('debug_train') 
        if 'debug_test' in kwargs                    :   debug_test = kwargs.pop('debug_test') 
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if dataroot:
            self.dataroot = dataroot
        if convo_layers_config:
            self.convo_layers_config = convo_layers_config
        if image_size:
            self.image_size = image_size
        if fc_layers_config:
            self.fc_layers_config = fc_layers_config
            if fc_layers_config[0] is not -1:
                raise Exception("""\n\n\nYour 'fc_layers_config' construction option is not correct. """
                                """The first element of the list of nodes in the fc layer must be -1 """
                                """because the input to fc will be set automatically to the size of """
                                """the final activation volume of the convolutional part of the network""")
        if  path_saved_model:
            self.path_saved_model = path_saved_model
        if classes:
            self.class_labels = classes
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if momentum:
            self.momentum = momentum
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                else:
                    raise Exception("You requested GPU support, but there's no GPU on this machine")
            else:
                self.device = torch.device("cpu")
        if debug_train:                             
            self.debug_train = debug_train
        else:
            self.debug_train = 0
        if debug_test:                             
            self.debug_test = debug_test
        else:
            self.debug_test = 0
        self.debug_config = 0
#        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu is False else "cpu")

    def parse_config_string_for_convo_layers(self):
        '''
        Each collection of 'n' otherwise identical layers in a convolutional network is 
        specified by a string that looks like:

                                 "nx[a,b,c,d]-MaxPool(k)"
        where 
                n      =  num of this type of convo layer
                a      =  number of out_channels                      [in_channels determined by prev layer] 
                b,c    =  kernel for this layer is of size (b,c)      [b along height, c along width]
                d      =  stride for convolutions
                k      =  maxpooling over kxk patches with stride of k

        Example:
                     "n1x[a1,b1,c1,d1]-MaxPool(k1)  n2x[a2,b2,c2,d2]-MaxPool(k2)"
        '''
        configuration = self.convo_layers_config
        configs = configuration.split()
        all_convo_layers = []
        image_size_after_layer = self.image_size
        for k,config in enumerate(configs):
            two_parts = config.split('-')
            how_many_conv_layers_with_this_config = int(two_parts[0][:config.index('x')])
            if self.debug_config:
                print("\n\nhow many convo layers with this config: %d" % how_many_conv_layers_with_this_config)
            maxpooling_size = int(re.findall(r'\d+', two_parts[1])[0])
            if self.debug_config:
                print("\nmax pooling size for all convo layers with this config: %d" % maxpooling_size)
            for conv_layer in range(how_many_conv_layers_with_this_config):            
                convo_layer = {'out_channels':None, 
                               'kernel_size':None, 
                               'convo_stride':None, 
                               'maxpool_size':None,
                               'maxpool_stride': None}
                kernel_params = two_parts[0][config.index('x')+1:][1:-1].split(',')
                if self.debug_config:
                    print("\nkernel_params: %s" % str(kernel_params))
                convo_layer['out_channels'] = int(kernel_params[0])
                convo_layer['kernel_size'] = (int(kernel_params[1]), int(kernel_params[2]))
                convo_layer['convo_stride'] =  int(kernel_params[3])
                image_size_after_layer = [x // convo_layer['convo_stride'] for x in image_size_after_layer]
                convo_layer['maxpool_size'] = maxpooling_size
                convo_layer['maxpool_stride'] = maxpooling_size
                image_size_after_layer = [x // convo_layer['maxpool_size'] for x in image_size_after_layer]
                all_convo_layers.append(convo_layer)
        configs_for_all_convo_layers = {i : all_convo_layers[i] for i in range(len(all_convo_layers))}
        if self.debug_config:
            print("\n\nAll convo layers: %s" % str(configs_for_all_convo_layers))
        last_convo_layer = configs_for_all_convo_layers[len(all_convo_layers)-1]
        out_nodes_final_layer = image_size_after_layer[0] * image_size_after_layer[1] * \
                                                                      last_convo_layer['out_channels']
        self.fc_layers_config[0] = out_nodes_final_layer
        self.configs_for_all_convo_layers = configs_for_all_convo_layers
        return configs_for_all_convo_layers


    def build_convo_layers(self, configs_for_all_convo_layers):
        conv_layers = nn.ModuleList()
        in_channels_for_next_layer = None
        for layer_index in configs_for_all_convo_layers:
            if self.debug_config:
                print("\n\n\nLayer index: %d" % layer_index)
            in_channels = 3 if layer_index == 0 else in_channels_for_next_layer
            out_channels = configs_for_all_convo_layers[layer_index]['out_channels']
            kernel_size = configs_for_all_convo_layers[layer_index]['kernel_size']
            padding = tuple((k-1) // 2 for k in kernel_size)
            stride       = configs_for_all_convo_layers[layer_index]['convo_stride']
            maxpool_size = configs_for_all_convo_layers[layer_index]['maxpool_size']
            if self.debug_config:
                print("\n     in_channels=%d   out_channels=%d    kernel_size=%s     stride=%s    \
                maxpool_size=%s" % (in_channels, out_channels, str(kernel_size), str(stride), 
                str(maxpool_size)))
            conv_layers.append( nn.Conv2d( in_channels,out_channels,kernel_size,stride=stride,padding=padding) )
            conv_layers.append( nn.MaxPool2d( maxpool_size ) )
            conv_layers.append( nn.ReLU() ),
            in_channels_for_next_layer = out_channels
        return conv_layers

    def build_fc_layers(self):
        fc_layers = nn.ModuleList()
        for layer_index in range(len(self.fc_layers_config) - 1):
            fc_layers.append( nn.Linear( self.fc_layers_config[layer_index], 
                                                                self.fc_layers_config[layer_index+1] ) )
        return fc_layers            

    def load_cifar_10_dataset(self):       
        '''
        We make sure that the transformation applied to the image end the images being normalized.
        Consider this call to normalize: "Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))".  The three
        numbers in the first tuple affect the means in the three color channels and the three 
        numbers in the second tuple affect the standard deviations.  In this case, we want the 
        image value in each channel to be changed to:

                 image_channel_val = (image_channel_val - mean) / std

        So with mean and std both set 0.5 for all three channels, if the image tensor originally 
        was between 0 and 1.0, after this normalization, the tensor will be between -1.0 and +1.0. 
        If needed we can do inverse normalization  by

                 image_channel_val  =   (image_channel_val * std) + mean
        '''
        ##   The call to ToTensor() converts the usual int range 0-255 for pixel values to 0-1.0 float vals
        ##   But then the call to Normalize() changes the range to -1.0-1.0 float vals.
        transform = tvt.Compose([tvt.ToTensor(),
                                 tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    ## accuracy: 51%
        ##  Define where the training and the test datasets are located:
        train_data_loc = torchvision.datasets.CIFAR10(root=self.dataroot, train=True,
                                                    download=True, transform=transform)
        test_data_loc = torchvision.datasets.CIFAR10(root=self.dataroot, train=False,
                                                    download=True, transform=transform)
        ##  Now create the data loaders:
        self.train_data_loader = torch.utils.data.DataLoader(train_data_loc,batch_size=self.batch_size,
                                                                            shuffle=True, num_workers=2)
        self.test_data_loader = torch.utils.data.DataLoader(test_data_loc,batch_size=self.batch_size,
                                                                           shuffle=False, num_workers=2)

    def load_cifar_10_dataset_with_augmentation(self):             
        '''
        In general, we want to do data augmentation for training:
        '''
        transform_train = tvt.Compose([
                                  tvt.RandomCrop(32, padding=4),
                                  tvt.RandomHorizontalFlip(),
                                  tvt.ToTensor(),
#                                  tvt.Normalize((0.20, 0.20, 0.20), (0.20, 0.20, 0.20))]) 
                                  tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])        
        ##  Don't need any augmentation for the test data: 
        transform_test = tvt.Compose([
                               tvt.ToTensor(),
#                               tvt.Normalize((0.20, 0.20, 0.20), (0.20, 0.20, 0.20))])  
                               tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ##  Define where the training and the test datasets are located
        train_data_loc = torchvision.datasets.CIFAR10(
                        root=self.dataroot, train=True, download=True, transform=transform_train)
        test_data_loc = torchvision.datasets.CIFAR10(
                      root=self.dataroot, train=False, download=True, transform=transform_test)
        ##  Now create the data loaders:
        self.train_data_loader = torch.utils.data.DataLoader(train_data_loc, batch_size=self.batch_size, 
                                                                     shuffle=True, num_workers=2)
        self.test_data_loader = torch.utils.data.DataLoader(test_data_loc, batch_size=self.batch_size, 
                                                                 shuffle=False, num_workers=2)

    def imshow(self, img):
        '''
        called by display_tensor_as_image() for displaying the image
        '''
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    class Net(nn.Module):
        def __init__(self, convo_layers, fc_layers):
            super(DLStudio.Net, self).__init__()
            self.my_modules_convo = convo_layers
            self.my_modules_fc = fc_layers
        def forward(self, x):
            for m in self.my_modules_convo:
                x = m(x)
            x = x.view(x.size(0), -1)
            for m in self.my_modules_fc:
                x = m(x)
            return x

    def show_network_summary(self, net):
        print("\n\n\nprinting out the model:")
        print(net)
        print("\n\n\na summary of input/output for the model:")
        summary(net, (3,self.image_size[0],self.image_size[1]),-1, device='cpu')

    def run_code_for_training(self, net):        
        filename_for_out = "performance_numbers_" + str(self.epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum)
        for epoch in range(self.epochs):  
            print("\n")
            running_loss = 0.0
            for i, data in enumerate(self.train_data_loader):
                inputs, labels = data
                if self.debug_train and i % 2000 == 1999:
                    print("\n\n[iter=%d:] Ground Truth:     " % (i+1) + 
                    ' '.join('%5s' % self.class_labels[labels[j]] for j in range(self.batch_size)))
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ##  Since PyTorch likes to construct dynamic computational graphs, we need to
                ##  zero out the previously calculated gradients for the learnable parameters:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                if self.debug_train and i % 2000 == 1999:
                    _, predicted = torch.max(outputs.data, 1)
                    print("[iter=%d:] Predicted Labels: " % (i+1) + 
                     ' '.join('%5s' % self.class_labels[predicted[j]] for j in range(self.batch_size)))
                    self.display_tensor_as_image(torchvision.utils.make_grid(inputs, normalize=True), 
                                            "see terminal for TRAINING results at iter=%d" % (i+1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:    
                    print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                    avg_loss = running_loss / float(2000)
                    print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
        print("\nFinished Training\n")
        self.save_model(net)


    def display_tensor_as_image(self, tensor, title=""):
        '''
        This method converts the argument tensor into a photo image that you can display
        in your terminal screen. It can convert tensors of three different shapes
        into images: (3,H,W), (1,H,W), and (H,W), where H, for height, stands for the
        number of pixels in the vertical direction and W, for width, for the same
        along the horizontal direction.  When the first element of the shape is 3,
        that means that the tensor represents a color image in which each pixel in
        the (H,W) plane has three values for the three color channels.  On the other
        hand, when the first element is 1, that stands for a tensor that will be
        shown as a grayscale image.  And when the shape is just (H,W), that is
        automatically taken to be for a grayscale image.
        '''
        tensor_range = (torch.min(tensor).item(), torch.max(tensor).item())
        if tensor_range == (-1.0,1.0):
            ##  The tensors must be between 0.0 and 1.0 for the display:
            print("\n\n\nimage un-normalization called")
            tensor = tensor/2.0 + 0.5     # unnormalize
        plt.figure(title)
        ###  The call to plt.imshow() shown below needs a numpy array. We must also
        ###  transpose the array so that the number of channels (the same thing as the
        ###  number of color planes) is in the last element.  For a tensor, it would be in
        ###  the first element.
        if tensor.shape[0] == 3 and len(tensor.shape) == 3:
#            plt.imshow( tensor.numpy().transpose(1,2,0) )
            plt.imshow( tensor.numpy().transpose(1,2,0) )
        ###  If the grayscale image was produced by calling torchvision.transform's
        ###  ".ToPILImage()", and the result converted to a tensor, the tensor shape will
        ###  again have three elements in it, however the first element that stands for
        ###  the number of channels will now be 1
        elif tensor.shape[0] == 1 and len(tensor.shape) == 3:
            tensor = tensor[0,:,:]
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        ###  For any one color channel extracted from the tensor representation of a color
        ###  image, the shape of the tensor will be (W,H):
        elif len(tensor.shape) == 2:
            plt.imshow( tensor.numpy(), cmap = 'gray' )
        else:
            sys.exit("\n\n\nfrom 'display_tensor_as_image()': tensor for image is ill formed -- aborting")
        plt.show()

    def check_a_sampling_of_images(self):
        '''
        Displays the first batch_size number of images in your dataset.
        '''
        dataiter = iter(self.train_data_loader)
        images, labels = dataiter.next()
        # Since negative pixel values make no sense for display, setting the 'normalize' 
        # option to True will change the range back from (-1.0,1.0) to (0.0,1.0):
        self.display_tensor_as_image(torchvision.utils.make_grid(images, normalize=True))
        # Print class labels for the images shown:
        print(' '.join('%5s' % self.class_labels[labels[j]] for j in range(self.batch_size)))

    def save_model(self, model):
        '''
        Save the trained model to a disk file
        '''
        torch.save(model.state_dict(), self.path_saved_model)

    def run_code_for_testing(self, net):
        net.load_state_dict(torch.load(self.path_saved_model))
        ##  In what follows, in addition to determining the predicted label for each test
        ##  image, we will also compute some stats to measure the overall performance of
        ##  the trained network.  This we will do in two different ways: For each class,
        ##  we will measure how frequently the network predicts the correct labels.  In
        ##  addition, we will compute the confusion matrix for the predictions.
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(len(self.class_labels), len(self.class_labels))
        class_correct = [0] * len(self.class_labels)
        class_total = [0] * len(self.class_labels)
        with torch.no_grad():
            for i,data in enumerate(self.test_data_loader):
                ##  data is set to the images and the labels for one batch at a time:
                images, labels = data
                if self.debug_test and i % 1000 == 0:
                    print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%5s' % self.class_labels[labels[j]] 
                                                               for j in range(self.batch_size)))
                outputs = net(images)
                ##  max() returns two things: the max value and its index in the 10 element
                ##  output vector.  We are only interested in the index --- since that is 
                ##  essentially the predicted class label:
                _, predicted = torch.max(outputs.data, 1)
                if self.debug_test and i % 1000 == 0:
                    print("[i=%d:] Predicted Labels: " %i + ' '.join('%5s' % self.class_labels[predicted[j]]
                                                              for j in range(self.batch_size)))
                    self.display_tensor_as_image(torchvision.utils.make_grid(images, normalize=True), 
                                                    "see terminal for test results at i=%d" % i)
                for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                ##  comp is a list of size batch_size of "True" and "False" vals
                comp = predicted == labels       
                for j in range(self.batch_size):
                    label = labels[j]
                    ##  The following works because, in a numeric context, the boolean value
                    ##  "False" is the same as number 0 and the boolean value True is the 
                    ##  same as number 1. For that reason "4 + True" will evaluate to 5 and
                    ##  "4 + False" will evaluate to 4.  Also, "1 == True" evaluates to "True"
                    ##  "1 == False" evaluates to "False".  However, note that "1 is True" 
                    ##  evaluates to "False" because the operator "is" does not provide a 
                    ##  numeric context for "True". And so on.  In the statement that follows,
                    ##  while  c[j].item() will either return "False" or "True", for the 
                    ##  addition operator, Python will use the values 0 and 1 instead.
                    class_correct[label] += comp[j].item()
                    class_total[label] += 1
        for j in range(len(self.class_labels)):
            print('Prediction accuracy for %5s : %2d %%' % (
                               self.class_labels[j], 100 * class_correct[j] / class_total[j]))
        print("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%" % 
                                                               (100 * correct / float(total)))
        print("\n\nDisplaying the confusion matrix:\n")
        out_str = "         "
        for j in range(len(self.class_labels)):  out_str +=  "%7s" % self.class_labels[j]   
        print(out_str + "\n")
        for i,label in enumerate(self.class_labels):
            out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                                      for j in range(len(self.class_labels))]
            out_percents = ["%.2f" % item.item() for item in out_percents]
            out_str = "%6s:  " % self.class_labels[i]
            for j in range(len(self.class_labels)): out_str +=  "%7s" % out_percents[j]
            print(out_str)


    ########################################################################################
    ###############  Start Definition of Inner Class ExperimentsWithSequential #############

    class ExperimentsWithSequential(nn.Module):                                
        """
        Demonstrates how to use the torch.nn.Sequential container class
        """
        def __init__(self, dl_studio ):
            super(DLStudio.ExperimentsWithSequential, self).__init__()
            self.dl_studio = dl_studio

        def load_cifar_10_dataset(self):       
            self.dl_studio.load_cifar_10_dataset()

        def load_cifar_10_dataset_with_augmentation(self):             
            self.dl_studio.load_cifar_10_dataset_with_augmentation()

        class Net(nn.Module):
            """
            To see if the DLStudio class would work with any network that a user may want
            to experiment with, I copy-and-pasted the network shown below from the following
            page by Zhenye at GitHub:
                         https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html
            """
            def __init__(self):
                super(DLStudio.ExperimentsWithSequential.Net, self).__init__()
                self.conv_seqn = nn.Sequential(
                    # Conv Layer block 1:
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    # Conv Layer block 2:
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(p=0.05),
                    # Conv Layer block 3:
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.fc_seqn = nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Linear(4096, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.1),
                    nn.Linear(512, 10)
                )
    
            def forward(self, x):
                x = self.conv_seqn(x)
                # flatten
                x = x.view(x.size(0), -1)
                x = self.fc_seqn(x)
                return x

        def run_code_for_training(self, net):        
            self.dl_studio.run_code_for_training(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing(self, model):
            self.dl_studio.run_code_for_testing(model)


    ########################################################################################
    ##################  Start Definition of Inner Class ExperimentsWithCIFAR ###############

    class ExperimentsWithCIFAR(nn.Module):              

        def __init__(self, dl_studio ):
            super(DLStudio.ExperimentsWithCIFAR, self).__init__()
            self.dl_studio = dl_studio

        def load_cifar_10_dataset(self):       
            self.dl_studio.load_cifar_10_dataset()

        def load_cifar_10_dataset_with_augmentation(self):             
            self.dl_studio.load_cifar_10_dataset_with_augmentation()

        ##  You can instantiate two different types of networks when experimenting with 
        ##  the inner class ExperimentsWithCIFAR.  The network shown below is from the 
        ##  PyTorch tutorial
        ##
        ##     https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        ##
        class Net(nn.Module):
            def __init__(self):
                super(DLStudio.ExperimentsWithCIFAR.Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
        
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        ##  Instead of using the network shown above, you can also use the network shown below.
        ##  if you are playing with the ExperimentsWithCIFAR inner class. If that's what you
        ##  want to do, in the script "playing_with_cifar10.py" in the Examples directory,
        ##  you will need to replace the statement
        ##                          model = exp_cifar.Net()
        ##  by the statement
        ##                          model = exp_cifar.Net2()        
        ##
        class Net2(nn.Module):
            def __init__(self):
                """
                I created this network class just to see if it was possible to simply calculate
                the size of the first of the fully connected layers from strides in the convo
                layers up to that point and from the out_channels used in the top-most convo 
                layer.   In what you see below, I am keeping track of all the strides by pushing 
                them into the array 'strides'.  Subsequently, in the formula shown in line (A),
                I use the product of all strides and the number of out_channels for the topmost
                layer to compute the size of the first fully-connected layer.
                """
                super(DLStudio.ExperimentsWithCIFAR.Net2, self).__init__()
                self.relu = nn.ReLU()
                strides = []
                patch_size = 2
                ## conv1:
                out_ch, ker_size, conv_stride, pool_stride = 128,5,1,2
                self.conv1 = nn.Conv2d(3, out_ch, (ker_size,ker_size), padding=(ker_size-1)//2)     
                self.pool1 = nn.MaxPool2d(patch_size, pool_stride)     
                strides += (conv_stride, pool_stride)
                ## conv2:
                in_ch = out_ch
                out_ch, ker_size, conv_stride, pool_stride = 128,3,1,2
                self.conv2 = nn.Conv2d(in_ch, out_ch, ker_size, padding=(ker_size-1)//2)
                self.pool2 = nn.MaxPool2d(patch_size, pool_stride)     
                strides += (conv_stride, pool_stride)
                ## conv3:                   
                ## meant for repeated invocation, must have same in_ch, out_ch and strides of 1
                in_ch = out_ch
                out_ch, ker_size, conv_stride, pool_stride = in_ch,2,1,1
                self.conv3 = nn.Conv2d(in_ch, out_ch, ker_size, padding=1)
                self.pool3 = nn.MaxPool2d(patch_size, pool_stride)         
#                strides += (conv_stride, pool_stride)
                ## figure out the number of nodes needed for entry into fc:
                in_size_for_fc = out_ch * (32 // np.prod(strides)) ** 2                    ## (A)
                self.in_size_for_fc = in_size_for_fc
                self.fc1 = nn.Linear(in_size_for_fc, 150)
                self.fc2 = nn.Linear(150, 100)
                self.fc3 = nn.Linear(100, 10)
        
            def forward(self, x):
                ##  We know that forward() begins its with work x shaped as (4,3,32,32) where
                ##  4 is the batch size, 3 in_channels, and where the input image size is 32x32.
                x = self.relu(self.conv1(x))  
                x = self.pool1(x)             
                x = self.relu(self.conv2(x))
                x = self.pool2(x)             
                for _ in range(5):
                    x = self.pool3(self.relu(self.conv3(x)))
                x = x.view(-1, self.in_size_for_fc)
                x = self.relu(self.fc1( x ))
                x = self.relu(self.fc2( x ))
                x = self.fc3(x)
                return x

        def run_code_for_training(self, net):        
            self.dl_studio.run_code_for_training(net)
            
        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing(self, model):
            self.dl_studio.run_code_for_testing(model)



    ########################################################################################
    ###############  Start Definition of Inner Class AutogradCustomization  ################

    class AutogradCustomization(nn.Module):             
        """
        This class illustrates how you can add additional functionality of Autograd by 
        following the instructions posted at
                   https://pytorch.org/docs/stable/notes/extending.html
        """

        def __init__(self, dl_studio, num_samples_per_class):
            super(DLStudio.AutogradCustomization, self).__init__()
            self.dl_studio = dl_studio
            self.num_samples_per_class = num_samples_per_class


        class DoSillyWithTensor(torch.autograd.Function):                  
            """        
            Extending Autograd requires that you define a new verb class, as I have with
            the class DoSillyWithTensor shown below, with definitions for two static
            methods, "forward()" and "backward()".  An instance constructed from this
            class is callable.  So when, in the "forward()" of the network, you pass a
            training sample through an instance of DoSillyWithTensor, it is subject to
            the code shown below in the "forward()"  of this class.
            """
            @staticmethod
            def forward(ctx, input):
                """
                The parameter 'input' is set to the training sample that is being 
                processed by an instance of DoSillyWithTensor in the "forward()" of a
                network.  We first make a deep copy of this tensor (which should be a 
                32-bit float) and then we subject the copy to a conversion to a one-byte 
                integer, which should cause a significant loss of information. We 
                calculate the difference between the original 32-bit float and the 8-bit 
                version and store it away in the context variable "ctx".
                """
                input_orig = input.clone().double()
                input = input.to(torch.uint8).double()
                diff = input_orig.sub(input)
                ctx.save_for_backward(diff)
                return input

            @staticmethod
            def backward(ctx, grad_output):
                """
                Whatever was stored in the context variable "ctx" during the forward pass
                can be retrieved in the backward pass as shown below.
                """
                diff, = ctx.saved_tensors
                grad_input = grad_output.clone()
                grad_input = grad_input + diff
                return grad_input
        
        def gen_training_data(self):        
            mean1,mean2   = [3.0,3.0], [5.0,5.0]
            covar1,covar2 = [[1.0,0.0], [0.0,1.0]], [[1.0,0.0], [0.0,1.0]]
            data1 = [(list(x),1) for x in np.random.multivariate_normal(mean1, covar1, 
                                                                     self.num_samples_per_class)]
            data2 = [(list(x),2) for x in np.random.multivariate_normal(mean2, covar2, 
                                                                     self.num_samples_per_class)]
            training_data = data1 + data2
            random.shuffle( training_data )
            self.training_data = training_data 

        def train_with_straight_autograd(self):
            dtype = torch.float
            D_in,H,D_out = 2,10,2
#           w1 = torch.randn(D_in, H, device="cpu", dtype=dtype, requires_grad=True)
#           w2 = torch.randn(H, D_out, device="cpu", dtype=dtype, requires_grad=True)
            w1 = torch.randn(D_in, H, device="cpu", dtype=dtype)
            w2 = torch.randn(H, D_out, device="cpu", dtype=dtype)
            w1 = w1.to(self.dl_studio.device)
            w2 = w2.to(self.dl_studio.device)
            w1.requires_grad_()
            w2.requires_grad_()
            Loss = []
            for epoch in range(self.dl_studio.epochs):
                for i,data in enumerate(self.training_data):
                    input, label = data
                    x,y = torch.as_tensor(np.array(input)), torch.as_tensor(np.array(label))
                    x,y = x.float(), y.float()
                    if self.dl_studio.use_gpu is True:
                        x,y = x.to(self.dl_studio.device), y.to(self.dl_studio.device)
                    y_pred = x.view(1,-1).mm(w1).clamp(min=0).mm(w2)
                    loss = (y_pred - y).pow(2).sum()
                    if i % 200 == 199:
                        Loss.append(loss.item())
                        print("epoch=%d i=%d" % (epoch,i), loss.item())
#                   w1.retain_grad()
#                   w2.retain_grad()
                    loss.backward()       
                    with torch.no_grad():
                        w1 -= self.dl_studio.learning_rate * w1.grad
                        w2 -= self.dl_studio.learning_rate * w2.grad
                        w1.grad.zero_()
                        w2.grad.zero_()
            print("\n\n\nLoss: %s" % str(Loss))
            import matplotlib.pyplot as plt
            plt.figure("Loss vs training (straight autograd)")
            plt.plot(Loss)
            plt.show()

        def train_with_extended_autograd(self):
            dtype = torch.float
            D_in,H,D_out = 2,10,2
#           w1 = torch.randn(D_in, H, device="cpu", dtype=dtype, requires_grad=True)
#           w2 = torch.randn(H, D_out, device="cpu", dtype=dtype, requires_grad=True)
            w1 = torch.randn(D_in, H, device="cpu", dtype=dtype)
            w2 = torch.randn(H, D_out, device="cpu", dtype=dtype)
            w1 = w1.to(self.dl_studio.device)
            w2 = w2.to(self.dl_studio.device)
            w1.requires_grad_()
            w2.requires_grad_()
            Loss = []
            for epoch in range(self.dl_studio.epochs):
                for i,data in enumerate(self.training_data):
                    ## Constructing an instance of DoSillyWithTensor. It is callable.
                    do_silly = DLStudio.AutogradCustomization.DoSillyWithTensor.apply      
                    input, label = data
                    x,y = torch.as_tensor(np.array(input)), torch.as_tensor(np.array(label))
                    ## Now process the training instance with the "do_silly" instance:
                    x = do_silly(x)                                 
                    x,y = x.float(), y.float()
                    x,y = x.to(self.dl_studio.device), y.to(self.dl_studio.device)
                    y_pred = x.view(1,-1).mm(w1).clamp(min=0).mm(w2)
                    loss = (y_pred - y).pow(2).sum()
                    if i % 200 == 199:
                        Loss.append(loss.item())
                        print("epoch=%d i=%d" % (epoch,i), loss.item())
#                   w1.retain_grad()
#                   w2.retain_grad()
                    loss.backward()       
                    with torch.no_grad():
                        w1 -= self.dl_studio.learning_rate * w1.grad
                        w2 -= self.dl_studio.learning_rate * w2.grad
                        w1.grad.zero_()
                        w2.grad.zero_()
            print("\n\n\nLoss: %s" % str(Loss))
            import matplotlib.pyplot as plt
            plt.figure("loss vs training (extended autograd)")
            plt.plot(Loss)
            plt.show()


    ########################################################################################
    ###################  Start Definition of Inner Class SkipConnections  ##################

    class SkipConnections(nn.Module):             
        """
        This educational class is meant for illustrating the concepts related to the 
        use of skip connections in neural network.  It is now well known that deep
        networks are difficult to train because of the vanishing gradients problem.
        What that means is that as the depth of network increases, the loss gradients
        calculated for the early layers become more and more muted, which suppresses
        the learning of the parameters in those layers.  An important mitigation
        strategy for addressing this problem consists of creating a CNN using blocks
        with skip connections.

        With the code shown in this inner class of the module, you can now experiment
        with skip connections in a CNN to see how a deep network with this feature
        might improve the classification results.  As you will see in the code shown
        below, the network that allows you to construct a CNN with skip connections
        is named BMEnet.  As shown in the script playing_with_skip_connections.py in
        the Examples directory of the distribution, you can easily create a CNN with
        arbitrary depth just by using the "depth" constructor option for the BMEnet
        class.  The basic block of the network constructed by BMEnet is called
        SkipBlock which, very much like the BasicBlock in ResNet-18, has a couple of
        convolutional layers whose output is combined with the input to the block.
    
        Note that the value given to the "depth" constructor option for the
        BMEnet class does NOT translate directly into the actual depth of the
        CNN. [Again, see the script playing_with_skip_connections.py in the Examples
        directory for how to use this option.] The value of "depth" is translated
        into how many instances of SkipBlock to use for constructing the CNN.
        """

        def load_cifar_10_dataset(self):       
            self.dl_studio.load_cifar_10_dataset()

        def load_cifar_10_dataset_with_augmentation(self):             
            self.dl_studio.load_cifar_10_dataset_with_augmentation()

        def __init__(self, dl_studio):
            super(DLStudio.SkipConnections, self).__init__()
            self.dl_studio = dl_studio

        class SkipBlock(nn.Module):
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.SkipConnections.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer = nn.BatchNorm2d
                self.bn = norm_layer(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

            def forward(self, x):
                identity = x                                     
                out = self.convo(x)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)                              
                    out = self.bn(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class BMEnet(nn.Module):
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.SkipConnections.BMEnet, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DLStudio.SkipConnections.SkipBlock(64, 64, skip_connections=skip_connections)
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                                downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                                skip_connections=skip_connections )
                self.skip128 = DLStudio.SkipConnections.SkipBlock(128, 128, skip_connections=skip_connections)
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                                downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 10)

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                for _ in range(self.depth // 4):
                    x = self.skip64(x)                                               
                x = self.skip64ds(x)
                for _ in range(self.depth // 4):
                    x = self.skip64(x)                                               
                x = self.skip64to128(x)
                for _ in range(self.depth // 4):
                    x = self.skip128(x)                                               
                x = self.skip128ds(x)                                               
                for _ in range(self.depth // 4):
                    x = self.skip128(x)                                               
                x = x.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x = torch.nn.functional.relu(self.fc1(x))
                x = self.fc2(x)
                return x            

        def run_code_for_training(self, net):        
            self.dl_studio.run_code_for_training(net)
            
        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing(self, model):
            self.dl_studio.run_code_for_testing(model)


    ########################################################################################
    #################  Start Definition of Inner Class CustomDataLoading  ##################

    class CustomDataLoading(nn.Module):             
        """This is a testbed for experimenting with a completely grounds-up attempt at
        designing a custom data loader.  Ordinarily, if the basic format of how the
        dataset is stored is similar to one of the datasets that the Torchvision
        module knows about, you can go ahead and use that for your own dataset.  At
        worst, you may need to carry out some light customizations depending on the
        number of classes involved, etc.

        However, if the underlying dataset is stored in a manner that does not look
        like anything in Torchvision, you have no choice but to supply yourself all
        of the data loading infrastructure.  That is what this inner class of the 
        DLStudio module is all about.

        The custom data loading exercise here is related to a dataset called
        PurdueShapes5 that contains 32x32 images of binary shapes belonging to the
        following five classes:

                       1.  rectangle
                       2.  triangle
                       3.  disk
                       4.  oval
                       5.  star

        The dataset was generated by randomizing the sizes and the orientations
        of these five patterns.  Since the patterns are rotated with a very simple
        non-interpolating transform, just the act of random rotations can introduce
        boundary and even interior noise in the patterns.

        Each 32x32 image is stored in the dataset as the following list:

                           [R, G, B, Bbox, Label]
        where
                R     :   is a 1024 element list of the values for the red component
                          of the color at all the pixels
           
                B     :   the same as above but for the green component of the color

                G     :   the same as above but for the blue component of the color

                Bbox  :   a list like [x1,y1,x2,y2] that defines the bounding box 
                          for the object in the image
           
                Label :   the shape of the object

        I serialize the dataset with Python's pickle module and then compress it with 
        the gzip module.  

        You will find the following dataset directories in the "data" subdirectory
        of Examples in the DLStudio distro:

               PurdueShapes5-10000-train.gz
               PurdueShapes5-1000-test.gz
               PurdueShapes5-20-train.gz
               PurdueShapes5-20-test.gz               

        The number that follows the main name string "PurdueShapes5-" is for the 
        number of images in the dataset.  

        You will find the last two datasets, with 20 images each, useful for debugging
        your logic for object detection and bounding-box regression.

        """     
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DLStudio.CustomDataLoading, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            def __init__(self, dl_studio, train_or_test, dataset_file, transform=None):
                super(DLStudio.CustomDataLoading.PurdueShapes5Dataset, self).__init__()
                if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
                    if os.path.exists("torch_saved_PurdueShapes5-10000_dataset.pt") and \
                              os.path.exists("torch_saved_PurdueShapes5_label_map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch_saved_PurdueShapes5-10000_dataset.pt")
                        self.label_map = torch.load("torch_saved_PurdueShapes5_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
#                        self.dataset, self.label_map = pickle.loads(dataset)
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        torch.save(self.dataset, "torch_saved_PurdueShapes5-10000_dataset.pt")
                        torch.save(self.label_map, "torch_saved_PurdueShapes5_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
#                    self.dataset, self.label_map = pickle.loads(dataset)
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
             
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
                im_tensor = torch.zeros(3,32,32, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                sample = {'image' : im_tensor, 
                          'bbox' : self.dataset[idx][3],                          
                          'label' : self.dataset[idx][4] }
                if self.transform:
                     sample = self.transform(sample)
                return sample

        def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       
            transform = tvt.Compose([tvt.ToTensor(),
                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                               batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=4)
    
        class SkipBlock(nn.Module):
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.CustomDataLoading.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer = nn.BatchNorm2d
                self.bn = norm_layer(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo(x)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)                              
                    out = self.bn(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class BMEnet(nn.Module):
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.CustomDataLoading.BMEnet, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = DLStudio.SkipConnections.SkipBlock(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 10)

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                for _ in range(self.depth // 4):
                    x = self.skip64(x)                                               
                x = self.skip64ds(x)
                for _ in range(self.depth // 4):
                    x = self.skip64(x)                                               
                x = self.skip64to128(x)
                for _ in range(self.depth // 4):
                    x = self.skip128(x)                                               
                x = self.skip128ds(x)                                               
                for _ in range(self.depth // 4):
                    x = self.skip128(x)                                               
                x = x.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x = torch.nn.functional.relu(self.fc1(x))
                x = self.fc2(x)
                return x            

        def run_code_for_training_with_custom_loading(self, net):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            for epoch in range(self.dl_studio.epochs):  
#                print("\n")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):
                    inputs, bounding_box, labels = data['image'], data['bbox'], data['label']
                    if self.dl_studio.debug_train and i % 1000 == 999:
                        print("\n\n\nlabels: %s" % str(labels))
                        print("\n\n\ntype of labels: %s" % type(labels))
                        print("\n\n[iter=%d:] Ground Truth:     " % (i+1) + 
                        ' '.join('%5s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    if self.dl_studio.debug_train and i % 1000 == 999:
                        _, predicted = torch.max(outputs.data, 1)
                        print("[iter=%d:] Predicted Labels: " % (i+1) + 
                         ' '.join('%5s' % self.dataserver.class_labels[predicted[j]] 
                                           for j in range(self.dl_studio.batch_size)))
                        self.dl_studio.display_tensor_as_image(torchvision.utils.make_grid(
             inputs, normalize=True), "see terminal for TRAINING results at iter=%d" % (i+1))
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    if i % 1000 == 999:    
                        avg_loss = running_loss / float(1000)
                        print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("\nFinished Training\n")
            self.save_model(net)
            
        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing_with_custom_loading(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                           len(self.dataserver_train.class_labels))
            class_correct = [0] * len(self.dataserver_train.class_labels)
            class_total = [0] * len(self.dataserver_train.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    images, bounding_box, labels = data['image'], data['bbox'], data['label']
                    labels = labels.tolist()
                    if self.dl_studio.debug_test and i % 1000 == 0:
                        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
    self.dataserver_train.class_labels[labels[j]] for j in range(self.dl_studio.batch_size)))
                    outputs = net(images)
                    ##  max() returns two things: the max value and its index in the 10 element
                    ##  output vector.  We are only interested in the index --- since that is 
                    ##  essentially the predicted class label:
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = predicted.tolist()
                    if self.dl_studio.debug_test and i % 1000 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
 self.dataserver_train.class_labels[predicted[j]] for j in range(self.dl_studio.batch_size)))
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(images, normalize=True), 
                              "see terminal for test results at i=%d" % i)
                    for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                    for j in range(self.dl_studio.batch_size):
                        label = labels[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.dataserver_train.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%" % 
                                                                   (100 * correct / float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                "
            for j in range(len(self.dataserver_train.class_labels)):  
                                 out_str +=  "%15s" % self.dataserver_train.class_labels[j]   
            print(out_str + "\n")
            for i,label in enumerate(self.dataserver_train.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.dataserver_train.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.dataserver_train.class_labels[i]
                for j in range(len(self.dataserver_train.class_labels)): 
                                                       out_str +=  "%15s" % out_percents[j]
                print(out_str)
    

    ########################################################################################
    ###################  Start Definition of Inner Class DetectAndLocalize  ################

    class DetectAndLocalize(nn.Module):             
        """
        The purpose of this inner class is to focus on object detection in images --- as
        opposed to image classification.  Most people would say that object detection
        is a more challenging problem than image classification because, in general,
        the former also requires localization.  The simplest interpretation of what
        is meant by localization is that the code that carries out object detection
        must also output a bounding-box rectangle for the object that was detected.

        You will find in this inner class some examples of LOADnet classes meant
        for solving the object detection and localization problem.  The acronym
        "LOAD" in "LOADnet" stands for

                    "LOcalization And Detection"

        The different network examples included here are LOADnet1, LOADnet2, and
        LOADnet3.  For now, only pay attention to LOADnet2 since that's the class I
        have worked with the most for the 1.0.7 distribution.
        """
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DLStudio.DetectAndLocalize, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class PurdueShapes5Dataset(torch.utils.data.Dataset):
            def __init__(self, dl_studio, train_or_test, dataset_file, transform=None):
                super(DLStudio.DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
                if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
                    if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                              os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                        print("\nLoading training data from the torch-saved archive")
                        self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                        self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a minute or so.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        if sys.version_info[0] == 3:
                            self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        else:
                            self.dataset, self.label_map = pickle.loads(dataset)
                        torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                        torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
             
            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
                im_tensor = torch.zeros(3,32,32, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
                sample = {'image' : im_tensor, 
                          'bbox' : bb_tensor,
                          'label' : self.dataset[idx][4] }
                if self.transform:
                     sample = self.transform(sample)
                return sample

        def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test ):       
#            transform = tvt.Compose([tvt.ToTensor(),
#                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                               batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=4)
    
        class SkipBlock(nn.Module):
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.DetectAndLocalize.SkipBlock, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer = nn.BatchNorm2d
                self.bn = norm_layer(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

            def forward(self, x):
                identity = x                                     
                out = self.convo(x)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)                              
                    out = self.bn(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class LOADnet1(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet1 only uses fully-connected layers for the regression
            """
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.DetectAndLocalize.LOADnet1, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = DLStudio.SkipConnections.SkipBlock(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 5)
                self.fc3 =  nn.Linear(32768, 1000)
                self.fc4 =  nn.Linear(1000, 4)


            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                ## The labeling section:
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x)                                               
                x1 = self.skip64ds(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x1)                                               
                x1 = self.skip64to128(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = self.skip128ds(x1)                                               
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                x2 = x.view(-1, 32768 )
                x2 = torch.nn.functional.relu(self.fc3(x2))
                x2 = self.fc4(x2)
                return x1,x2

        class LOADnet2(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet2 uses both convo and linear layers for regression
            """ 
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.DetectAndLocalize.LOADnet2, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = DLStudio.SkipConnections.SkipBlock(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 5)
                ##  for regression
                self.conv_seqn = nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
                self.fc_seqn = nn.Sequential(
                    nn.Linear(16384, 1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
                )

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                ## The labeling section:
                x1 = x.clone()
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x1)                                               
                x1 = self.skip64ds(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x1)                                               
                x1 = self.skip64to128(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = self.skip128ds(x1)                                               
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                x2 = self.conv_seqn(x)
                x2 = self.conv_seqn(x2)
                # flatten
                x2 = x2.view(x.size(0), -1)
                x2 = self.fc_seqn(x2)
                return x1,x2

        class LOADnet3(nn.Module):
            """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet2 uses both convo and linear layers for regression
            """ 
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.DetectAndLocalize.LOADnet2, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.skip64 = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64ds = DLStudio.SkipConnections.SkipBlock(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128 = DLStudio.SkipConnections.SkipBlock(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128 = DLStudio.SkipConnections.SkipBlock(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128ds = DLStudio.SkipConnections.SkipBlock(128,128,
                                            downsample=True, skip_connections=skip_connections)
                self.fc1 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc2 =  nn.Linear(1000, 5)
                self.fc3 =  nn.Linear(128 * (32 // 2**self.pool_count)**2, 1000)
                self.fc4 =  nn.Linear(2048, 1024)

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv(x)))          
                ## The labeling section:
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x)                                               
                x1 = self.skip64ds(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64(x1)                                               
                x1 = self.skip64to128(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = self.skip128ds(x1)                                               
                for _ in range(self.depth // 4):
                    x1 = self.skip128(x1)                                               
                x1 = x1.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x1 = torch.nn.functional.relu(self.fc1(x1))
                x1 = self.fc2(x1)
                ## The Bounding Box regression:
                for _ in range(4):
                    x2 = self.skip64(x)                                               
                x2 = self.skip64to128(x2)
                for _ in range(4):
                    x2 = self.skip128(x2)                                               

                x2 = x.view(-1, 128 * (32 // 2**self.pool_count)**2 )
                x2 = torch.nn.functional.relu(self.fc3(x2))
                x2 = self.fc4(x2)
                return x1,x2

        class IOULoss(nn.Module):
            def __init__(self, batch_size):
                super(DLStudio.DetectAndLocalize.IOULoss, self).__init__()
                self.batch_size = batch_size
            def forward(self, input, target):
                composite_loss = []
                for idx in range(self.batch_size):
                    union = intersection = 0.0
                    for i in range(32):
                        for j in range(32):
                            inp = input[idx,i,j]
                            tap = target[idx,i,j]
                            if (inp == tap) and (inp==1):
                                intersection += 1
                                union += 1
                            elif (inp != tap) and ((inp==1) or (tap==1)):
                                union += 1
                    if union == 0.0:
                        raise Exception("something_wrong")
                    batch_sample_iou = intersection / float(union)
                    composite_loss.append(batch_sample_iou)
                total_iou_for_batch = sum(composite_loss) 
                return 1 - torch.tensor([total_iou_for_batch / self.batch_size])

        def run_code_for_training_with_CrossEntropy_and_BCE_Losses(self, net):        
            """
            BCE stands for the Binary Cross Entropy Loss that is used for
            the regression loss in this training method.
            """
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.CrossEntropyLoss()
#            criterion2 = self.dl_studio.DetectAndLocalize.IOULoss(self.dl_studio.batch_size)
            criterion2 = nn.BCELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            for epoch in range(self.dl_studio.epochs):  
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if self.dl_studio.debug_train and i % 1000 == 999:
                        print("\n\n[iter=%d:] Ground Truth:     " % (i+1) + 
                        ' '.join('%5s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    bbox_gt = bbox_gt.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    if self.dl_studio.debug_train and i % 500 == 499:
                        inputs_copy = inputs.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>31] = 31
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[iter=%d:] Predicted Labels: " % (i+1) + 
                         ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] 
                                           for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(inputs_copy, normalize=True),
                             "see terminal for TRAINING results at iter=%d" % (i+1))
                    mask_regress = torch.zeros(self.dl_studio.batch_size,32,32,requires_grad=False)
                    mask_gt = torch.zeros(self.dl_studio.batch_size, 32,32)
                    for k,out_regres in enumerate(bbox_pred):
                        x1,y1,x2,y2 = bbox_pred[k].tolist()
                        x1_gt,y1_gt,x2_gt,y2_gt = bbox_gt[k].tolist()
                        x1,y1,x2,y2 = [int(item) if item >0 else 0 for item in (x1,y1,x2,y2)]
                        x1_gt,y1_gt,x2_gt,y2_gt = [int(item) if item>0 else 0 for item in (x1_gt,y1_gt,x2_gt,y2_gt)]
                        if abs(x1_gt - x2_gt)<5 or abs(y1_gt-y2_gt) < 5: gt_too_small = True
                        mask_regress_np = np.zeros((32,32), dtype=bool)
                        mask_gt_np = np.zeros((32,32), dtype=bool)
                        mask_regress_np[y1:y2,x1:x2] = 1
                        mask_gt_np[y1_gt:y2_gt, x1_gt:x2_gt] = 1
                        mask_regress[k,:,:] = torch.from_numpy(mask_regress_np)
                        mask_regress.reqiures_grad=True
                        mask_gt[k,:,:] = torch.from_numpy(mask_gt_np)
                        mask_gt.reqiures_grad=True                
                    loss_labeling = criterion1(outputs_label, labels)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(mask_regress, mask_gt)
                    loss_regression.requires_grad = True
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 1000 == 999:    
                        avg_loss_labeling = running_loss_labeling / float(1000)
                        avg_loss_regression = running_loss_regression / float(1000)
                        print("[epoch:%d, batch:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
            print("\nFinished Training\n")
            self.save_model(net)

        def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
            filename_for_out2 = "performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
            FILE1 = open(filename_for_out1, 'w')
            FILE2 = open(filename_for_out2, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            for epoch in range(self.dl_studio.epochs):  
                running_loss_labeling = 0.0
                running_loss_regression = 0.0       
                for i, data in enumerate(self.train_dataloader):
                    gt_too_small = False
                    inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                    if self.dl_studio.debug_train and i % 500 == 499:
#                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                        print("\n\n[epoch=%d iter=%d:] Ground Truth:     " % (epoch+1, i+1) + 
                        ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] for j in range(self.dl_studio.batch_size)))
                    inputs = inputs.to(self.dl_studio.device)
                    labels = labels.to(self.dl_studio.device)
                    bbox_gt = bbox_gt.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    outputs_label = outputs[0]
                    bbox_pred = outputs[1]
                    if self.dl_studio.debug_train and i % 500 == 499:
#                  if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                        inputs_copy = inputs.detach().clone()
                        inputs_copy = inputs_copy.cpu()
                        bbox_pc = bbox_pred.detach().clone()
                        bbox_pc[bbox_pc<0] = 0
                        bbox_pc[bbox_pc>31] = 31
                        bbox_pc[torch.isnan(bbox_pc)] = 0
                        _, predicted = torch.max(outputs_label.data, 1)
                        print("[epoch=%d iter=%d:] Predicted Labels: " % (epoch+1, i+1) + 
                         ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()] 
                                           for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bbox_gt[idx][1])
                            i2 = int(bbox_gt[idx][3])
                            j1 = int(bbox_gt[idx][0])
                            j2 = int(bbox_gt[idx][2])
                            k1 = int(bbox_pc[idx][1])
                            k2 = int(bbox_pc[idx][3])
                            l1 = int(bbox_pc[idx][0])
                            l2 = int(bbox_pc[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            inputs_copy[idx,0,i1:i2,j1] = 255
                            inputs_copy[idx,0,i1:i2,j2] = 255
                            inputs_copy[idx,0,i1,j1:j2] = 255
                            inputs_copy[idx,0,i2,j1:j2] = 255
                            inputs_copy[idx,2,k1:k2,l1] = 255                      
                            inputs_copy[idx,2,k1:k2,l2] = 255
                            inputs_copy[idx,2,k1,l1:l2] = 255
                            inputs_copy[idx,2,k2,l1:l2] = 255
#                        self.dl_studio.display_tensor_as_image(
#                              torchvision.utils.make_grid(inputs_copy, normalize=True),
#                             "see terminal for TRAINING results at iter=%d" % (i+1))
                    loss_labeling = criterion1(outputs_label, labels)
                    loss_labeling.backward(retain_graph=True)        
                    loss_regression = criterion2(bbox_pred, bbox_gt)
                    loss_regression.backward()
                    optimizer.step()
                    running_loss_labeling += loss_labeling.item()    
                    running_loss_regression += loss_regression.item()                
                    if i % 500 == 499:    
                        avg_loss_labeling = running_loss_labeling / float(500)
                        avg_loss_regression = running_loss_regression / float(500)
                        print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                        FILE1.write("%.3f\n" % avg_loss_labeling)
                        FILE1.flush()
                        FILE2.write("%.3f\n" % avg_loss_regression)
                        FILE2.flush()
                        running_loss_labeling = 0.0
                        running_loss_regression = 0.0
                    if self.dl_studio.debug_train and i%500==499:
#                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(inputs_copy, normalize=True),
                             "see terminal for TRAINING results at iter=%d" % (i+1))


            print("\nFinished Training\n")
            self.save_model(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing_detection_and_localization(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            correct = 0
            total = 0
            confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels), 
                                           len(self.dataserver_train.class_labels))
            class_correct = [0] * len(self.dataserver_train.class_labels)
            class_total = [0] * len(self.dataserver_train.class_labels)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    images, bounding_box, labels = data['image'], data['bbox'], data['label']
                    labels = labels.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
    self.dataserver_train.class_labels[labels[j]] for j in range(self.dl_studio.batch_size)))
                    outputs = net(images)
                    outputs_label = outputs[0]
                    outputs_regression = outputs[1]
                    outputs_regression[outputs_regression < 0] = 0
                    outputs_regression[outputs_regression > 31] = 31
                    outputs_regression[torch.isnan(outputs_regression)] = 0
                    output_bb = outputs_regression.tolist()
                    _, predicted = torch.max(outputs_label.data, 1)
                    predicted = predicted.tolist()
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
 self.dataserver_train.class_labels[predicted[j]] for j in range(self.dl_studio.batch_size)))
                        for idx in range(self.dl_studio.batch_size):
                            i1 = int(bounding_box[idx][1])
                            i2 = int(bounding_box[idx][3])
                            j1 = int(bounding_box[idx][0])
                            j2 = int(bounding_box[idx][2])
                            k1 = int(output_bb[idx][1])
                            k2 = int(output_bb[idx][3])
                            l1 = int(output_bb[idx][0])
                            l2 = int(output_bb[idx][2])
                            print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                            print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                            images[idx,0,i1:i2,j1] = 255
                            images[idx,0,i1:i2,j2] = 255
                            images[idx,0,i1,j1:j2] = 255
                            images[idx,0,i2,j1:j2] = 255
                            images[idx,2,k1:k2,l1] = 255                      
                            images[idx,2,k1:k2,l2] = 255
                            images[idx,2,k1,l1:l2] = 255
                            images[idx,2,k2,l1:l2] = 255
                        self.dl_studio.display_tensor_as_image(
                              torchvision.utils.make_grid(images, normalize=True), 
                              "see terminal for test results at i=%d" % i)
                    for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                    total += len(labels)
                    correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                    comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                    for j in range(self.dl_studio.batch_size):
                        label = labels[j]
                        class_correct[label] += comp[j]
                        class_total[label] += 1
            print("\n")
            for j in range(len(self.dataserver_train.class_labels)):
                print('Prediction accuracy for %5s : %2d %%' % (
              self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
            print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                                   (100 * correct / float(total)))
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                "
            for j in range(len(self.dataserver_train.class_labels)):  
                                 out_str +=  "%15s" % self.dataserver_train.class_labels[j]   
            print(out_str + "\n")
            for i,label in enumerate(self.dataserver_train.class_labels):
                out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                 for j in range(len(self.dataserver_train.class_labels))]
                out_percents = ["%.2f" % item.item() for item in out_percents]
                out_str = "%12s:  " % self.dataserver_train.class_labels[i]
                for j in range(len(self.dataserver_train.class_labels)): 
                                                       out_str +=  "%15s" % out_percents[j]
                print(out_str)


    ########################################################################################
    ##################  Start Definition of Inner Class SemanticSegmentation  ##############

    class SemanticSegmentation(nn.Module):             
        """The purpose of this inner class is to be able to use the DLStudio module for
           experiments with semantic segmentation.  At its simplest level, the
           purpose of semantic segmentation is to assign correct labels to the
           different objects in a scene, while localizing them at the same time.  At
           a more sophisticated level, a system that carries out semantic
           segmentation should also output a symbolic expression based on the objects
           found in the image and their spatial relationships with one another.

           The workhorse of this inner class is the mUnet network that is based
           on the UNET network that was first proposed by Ronneberger, Fischer and
           Brox in the paper "U-Net: Convolutional Networks for Biomedical Image
           Segmentation".  Their Unet extracts binary masks for the cell pixel blobs
           of interest in biomedical images.  The output of their Unet can
           therefore be treated as a pixel-wise binary classifier at each pixel
           position.  The mUnet class, on the other hand, is intended for
           segmenting out multiple objects simultaneously form an image. [A weaker
           reason for "Multi" in the name of the class is that it uses skip
           connections not only across the two arms of the "U", but also also along
           the arms.  The skip connections in the original Unet are only between the
           two arms of the U.  In mUnet, each object type is assigned a separate
           channel in the output of the network.

           This version of DLStudio also comes with a new dataset,
           PurdueShapes5MultiObject, for experimenting with mUnet.  Each image in
           this dataset contains a random number of selections from five different
           shapes, with the shapes being randomly scaled, oriented, and located in
           each image.  The five different shapes are: rectangle, triangle, disk,
           oval, and star.
        """
        def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(DLStudio.SemanticSegmentation, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class PurdueShapes5MultiObjectDataset(torch.utils.data.Dataset):
            """
            The very first thing to note is that the images in the dataset
            PurdueShapes5MultiObjectDataset are of size 64x64.  Each image has a
            random number (up to five) of the objects drawn from the following five
            shapes: rectangle, triangle, disk, oval, and star.  Each shape is
            randomized with respect to all its parameters, including those for its
            scale and location in the image.

            Each image in the dataset is represented by two data objects, one a list
            and the other a dictionary. The list data objects consists of the
            following items:

                [R, G, B, mask_array, mask_val_to_bbox_map]                                   ## (A)
            
            and the other data object is a dictionary that is set to:
            
                label_map = {'rectangle':50, 
                             'triangle' :100, 
                             'disk'     :150, 
                             'oval'     :200, 
                             'star'     :250}                                                 ## (B)
            
            Note that that second data object for each image is the same, as shown
            above.

            In the rest of this comment block, I'll explain in greater detail the
            elements of the list in line (A) above.

            
            R,G,B:
            ------

            Each of these is a 4096-element array whose elements store the
            corresponding color values at each of the 4096 pixels in a 64x64 image.
            That is, R is a list of 4096 integers, each between 0 and 255, for the
            value of the red component of the color at each pixel. Similarly, for G
            and B.
            

            mask_array:
            ----------

            The fourth item in the list shown in line (A) above is for the mask which is
            a numpy array of shape:
            
                           (5, 64, 64)
            
            It is initialized by the command:
            
                 mask_array = np.zeros((5,64,64), dtype=np.uint8)
            
            In essence, the mask_array consists of five planes, each of size 64x64.
            Each plane of the mask array represents an object type according to the
            following shape_index
            
                    shape_index = (label_map[shape] - 50) // 50
            
            where the label_map is as shown in line (B) above.  In other words, the
            shape_index values for the different shapes are:
            
                     rectangle:  0
                      triangle:  1
                          disk:  2
                          oval:  3
                          star:  4
            
            Therefore, the first layer (of index 0) of the mask is where the pixel
            values of 50 are stored at all those pixels that belong to the rectangle
            shapes.  Similarly, the second mask layer (of index 1) is where the pixel
            values of 100 are stored at all those pixel coordinates that belong to
            the triangle shapes in an image; and so on.
            
            It is in the manner described above that we define five different masks
            for an image in the dataset.  Each mask is for a different shape and the
            pixel values at the nonzero pixels in each mask layer are keyed to the
            shapes also.
            
            A reader is likely to wonder as to the need for this redundancy in the
            dataset representation of the shapes in each image.  Such a reader is
            likely to ask: Why can't we just use the binary values 1s and 0s in each
            mask layer where the corresponding pixels are in the image?  Setting
            these mask values to 50, 100, etc., was done merely for convenience.  I
            went with the intuition that the learning needed for multi-object
            segmentation would become easier if each shape was represented by a
            different pixels value in the corresponding mask. So I went ahead
            incorporated that in the dataset generation program itself.

            The mask values for the shapes are not to be confused with the actual RGB
            values of the pixels that belong to the shapes. The RGB values at the
            pixels in a shape are randomly generated.  Yes, all the pixels in a shape
            instance in an image have the same RGB values (but that value has nothing
            to do with the values given to the mask pixels for that shape).
            
            
            mask_val_to_bbox_map:
            --------------------
                   
            The fifth item in the list in line (A) above is a dictionary that tells us
            what bounding-box rectangle to associate with each shape in the image.  To
            illustrate what this dictionary looks like, assume that an image contains
            only one rectangle and only one disk, the dictionary in this case will look
            like:
            
                mask values to bbox mappings:  {200: [], 
                                                250: [], 
                                                100: [], 
                                                 50: [[56, 20, 63, 25]], 
                                                150: [[37, 41, 55, 59]]}
            
            Should there happen to be two rectangles in the same image, the dictionary
            would then be like:
            
                mask values to bbox mappings:  {200: [], 
                                                250: [], 
                                                100: [], 
                                                 50: [[56, 20, 63, 25], [18, 16, 32, 36]], 
                                                150: [[37, 41, 55, 59]]}
            
            Therefore, it is not a problem even if all the objects in an image are of
            the same type.  Remember, the object that are selected for an image are
            shown randomly from the different shapes.  By the way, an entry like '[56,
            20, 63, 25]' for the bounding box means that the upper-left corner of the
            BBox for the 'rectangle' shape is at (56,20) and the lower-right corner of
            the same is at the pixel coordinates (63,25).
            
            As far as the BBox quadruples are concerned, in the definition
            
                    [min_x,min_y,max_x,max_y]
            
            note that x is the horizontal coordinate, increasing to the right on your
            screen, and y is the vertical coordinate increasing downwards.

            """
            def __init__(self, dl_studio, train_or_test, dataset_file, transform=None):
                super(DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset, self).__init__()
                if train_or_test == 'train' and dataset_file == "PurdueShapes5MultiObject-10000-train.gz":
                    if os.path.exists("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt") and \
                              os.path.exists("torch_saved_PurdueShapes5MultiObject_label_map.pt"):
                        print("\nLoading training data from torch saved file")
                        self.dataset = torch.load("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        self.label_map = torch.load("torch_saved_PurdueShapes5MultiObject_label_map.pt")
                        self.transform = transform
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """up to 3 minutes.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        torch.save(self.dataset, "torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        torch.save(self.label_map, "torch_saved_PurdueShapes5MultiObject_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.transform = transform
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(64,64), g.reshape(64,64), b.reshape(64,64)
                im_tensor = torch.zeros(3,64,64, dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                mask_array = self.dataset[idx][3]
                mask_tensor = torch.from_numpy(mask_array)
                mask_val_to_bbox_map =  self.dataset[idx][4]
                max_bboxes_per_entry_in_map = max([ len(mask_val_to_bbox_map[key]) for key in mask_val_to_bbox_map ])
                ##  The first arg 5 is for the number of bboxes we are going to need. If all the
                ##  shapes are exactly the same, you are going to need five different bbox'es.
                ##  The second arg is the index reserved for each shape in a single bbox
                bbox_tensor = torch.zeros(5,5,4, dtype=torch.float)
                for bbox_idx in range(max_bboxes_per_entry_in_map):
                    for key in mask_val_to_bbox_map:
                        if len(mask_val_to_bbox_map[key]) == 1:
                            if bbox_idx == 0:
                                bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                        elif len(mask_val_to_bbox_map[key]) > 1 and bbox_idx < len(mask_val_to_bbox_map[key]):
                            bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                sample = {'image'        : im_tensor, 
                          'mask_tensor'  : mask_tensor,
                          'bbox_tensor'  : bbox_tensor }
                return sample

        def load_PurdueShapes5MultiObject_dataset(self, dataserver_train, dataserver_test ):   
#            transform = tvt.Compose([tvt.ToTensor(),
#                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=0)

        class SkipBlockDN(nn.Module):
            """
            This class for the skip connections in the downward leg of the "U"
            """
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(DLStudio.SemanticSegmentation.SkipBlockDN, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                norm_layer = nn.BatchNorm2d
                self.bn = norm_layer(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo(x)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)                              
                    out = self.bn(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
                return out

        class SkipBlockUP(nn.Module):
            """
            This class is for the skip connections in the upward leg of the "U"
            """
            def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
                super(DLStudio.SemanticSegmentation.SkipBlockUP, self).__init__()
                self.upsample = upsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convoT = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                norm_layer = nn.BatchNorm2d
                self.bn = norm_layer(out_ch)
                if upsample:
                    self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)
            def forward(self, x):
                identity = x                                     
                out = self.convoT(x)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convoT(out)                              
                    out = self.bn(out)                              
                    out = torch.nn.functional.relu(out)
                if self.upsample:
                    out = self.upsampler(out)
                    identity = self.upsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out += identity[:,self.out_ch:,:,:]
                return out


        class mUnet(nn.Module):
            """
            This network is called mUnet because it is intended for segmenting
            out multiple objects simultaneously form an image. [A weaker reason for
            "Multi" in the name of the class is that it uses skip connections not
            only across the two arms of the "U", but also also along the arms.]  The
            classic UNET was first proposed by Ronneberger, Fischer and Brox in the
            paper "U-Net: Convolutional Networks for Biomedical Image Segmentation".
            Their UNET extracts binary masks for the cell pixel blobs of interest
            in biomedical images.  The output of their UNET therefore can therefore
            be treated as a pixel-wise binary classifier at each pixel position.  

            The mUnet presented here, on the other hand, is meant specifically
            for simultaneously identifying and localizing multiple objects in a
            given image.  Each object type is assigned a separate channel in the
            output of the network.  

            I have created a dataset, PurdueShapes5MultiObject, for experimenting
            with mUnet.  Each image in this dataset contains a random number of
            selections from five different shapes, with the shapes being randomly
            scaled, oriented, and located in each image.  The five different shapes
            are: rectangle, triangle, disk, oval, and star.
            """ 
            def __init__(self, skip_connections=True, depth=32):
                super(DLStudio.SemanticSegmentation.mUnet, self).__init__()
                self.pool_count = 3
                self.depth = depth // 2
                self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                ##  For the DN arm of the U:
                self.skip64DN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 64, 
                                           downsample=True, skip_connections=skip_connections)
                self.skip64to128DN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 128, 
                                                            skip_connections=skip_connections )
                self.skip128DN = DLStudio.SemanticSegmentation.SkipBlockDN(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(128,128,
                                            downsample=True, skip_connections=skip_connections)
                ##  For the UP arm of the U:
                self.skip64UP = DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, 
                                                           skip_connections=skip_connections)
                self.skip64usUP = DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, 
                                           upsample=True, skip_connections=skip_connections)
                self.skip128to64UP = DLStudio.SemanticSegmentation.SkipBlockUP(128, 64, 
                                                            skip_connections=skip_connections )
                self.skip128UP = DLStudio.SemanticSegmentation.SkipBlockUP(128, 128, 
                                                             skip_connections=skip_connections)
                self.skip128usUP = DLStudio.SemanticSegmentation.SkipBlockUP(128,128,
                                            upsample=True, skip_connections=skip_connections)
                self.conv_out = nn.ConvTranspose2d(64, 5, 3, stride=2,dilation=2,output_padding=1,padding=2)

            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv_in(x)))          
                ## Going down to the bottom of U:
                x1 = x.clone()
                for _ in range(self.depth // 4):
                    x1 = self.skip64DN(x1)                                               
                num_channels_to_save1 = x1.shape[1] // 2
                save_for_upside_1 = x1[:,:num_channels_to_save1,:,:].clone()
                x1 = self.skip64dsDN(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64DN(x1)                                               
                num_channels_to_save2 = x1.shape[1] // 2
                save_for_upside_2 = x1[:,:num_channels_to_save2,:,:].clone()
                x1 = self.skip64to128DN(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip128DN(x1)                                               
                num_channels_to_save3 = x1.shape[1] // 2
                save_for_upside_3 = x1[:,:num_channels_to_save3,:,:].clone()
                x1 = self.skip128dsDN(x1)                                               
                for _ in range(self.depth // 4):
                    x1 = self.skip128DN(x1)                                               
                ## Coming up from the bottom of U on the other side:
                for _ in range(self.depth // 4):
                    x1 = self.skip128UP(x1)                                               
                x1 = self.skip128usUP(x1)          
                for _ in range(self.depth // 4):
                    x1 = self.skip128UP(x1)                                               
                x1[:,:num_channels_to_save3,:,:] =  save_for_upside_3
                x1 = self.skip128to64UP(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64UP(x1)                                               
                x1[:,:num_channels_to_save2,:,:] =  save_for_upside_2
                x1 = self.skip64usUP(x1)
                for _ in range(self.depth // 4):
                    x1 = self.skip64UP(x1) 
                x1[:,:num_channels_to_save1,:,:] =  save_for_upside_1
                x1 = self.conv_out(x1)
                return x1

        class SegmentationLoss(nn.Module):
            """
            I wrote this class before I switched to MSE loss.  I am leaving it here
            in case I need to get back to it in the future.  
            """
            def __init__(self, batch_size):
                super(DLStudio.SemanticSegmentation.SegmentationLoss, self).__init__()
                self.batch_size = batch_size
            def forward(self, output, mask_tensor):
                composite_loss = torch.zeros(1,self.batch_size)
                mask_based_loss = torch.zeros(1,5)
                for idx in range(self.batch_size):
                    outputh = output[idx,0,:,:]
                    for mask_layer_idx in range(mask_tensor.shape[0]):
                        mask = mask_tensor[idx,mask_layer_idx,:,:]
                        element_wise = (outputh - mask)**2                   
                        mask_based_loss[0,mask_layer_idx] = torch.mean(element_wise)
                    composite_loss[0,idx] = torch.sum(mask_based_loss)
                return torch.sum(composite_loss) / self.batch_size

        def run_code_for_training_for_semantic_segmentation(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion1 = nn.MSELoss()
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_segmentation = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    im_tensor   = im_tensor.to(self.dl_studio.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.dl_studio.device)                 
                    bbox_tensor = bbox_tensor.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    output = net(im_tensor) 
                    segmentation_loss = criterion1(output, mask_tensor)
                    segmentation_loss.backward()
                    optimizer.step()
                    running_loss_segmentation += segmentation_loss.item()    
#                    if (epoch==0 and (i==99 or i==499)) or (i%1000==999):    
                    if i%1000==999:    
                        avg_loss_segmentation = running_loss_segmentation / float(1000)
                        print("[epoch:%d,batch:%5d]  MSE loss: %.3f" % (epoch+1,i+1,avg_loss_segmentation))
                        FILE1.write("%.3f\n" % avg_loss_segmentation)
                        FILE1.flush()
                        running_loss_segmentation = 0.0
            print("\nFinished Training\n")
            self.save_model(net)

        def save_model(self, model):
            '''
            Save the trained model to a disk file
            '''
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)

        def run_code_for_testing_semantic_segmentation(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    if self.dl_studio.debug_test and i % 50 == 0:
                        print("\n\n\n\nShowing output for test batch %d: " % (i+1))
                        outputs = net(im_tensor)                        
                        ## In the statement below: 1st arg for batch items, 2nd for channels, 
                        ##                         3rd and 4th for image size
                        output_bw_tensor = torch.zeros(4,1,64,64, dtype=float)
                        for image_idx in range(self.dl_studio.batch_size):
                            for layer_idx in range(5):
                                for m in range(64):
                                    for n in range(64):
                                        output_bw_tensor[image_idx,0,m,n]  =  \
                                                  torch.max( outputs[image_idx,:,m,n] )
#                        display_tensor = torch.zeros(8,3,64,64, dtype=float)
                        display_tensor = torch.zeros(28,3,64,64, dtype=float)
                        for idx in range(self.dl_studio.batch_size):
                            for bbox_idx in range(5):         ## 5 for the five different types of obj
                                bb_tensor = bbox_tensor[idx,bbox_idx]
                                for k in range(5):
                                    i1 = int(bb_tensor[k][1])
                                    i2 = int(bb_tensor[k][3])
                                    j1 = int(bb_tensor[k][0])
                                    j2 = int(bb_tensor[k][2])
                                    output_bw_tensor[idx,0,i1:i2,j1] = 255
                                    output_bw_tensor[idx,0,i1:i2,j2] = 255
                                    output_bw_tensor[idx,0,i1,j1:j2] = 255
                                    output_bw_tensor[idx,0,i2,j1:j2] = 255
                                    im_tensor[idx,0,i1:i2,j1] = 255
                                    im_tensor[idx,0,i1:i2,j2] = 255
                                    im_tensor[idx,0,i1,j1:j2] = 255
                                    im_tensor[idx,0,i2,j1:j2] = 255
                        display_tensor[:4,:,:,:] = output_bw_tensor
                        display_tensor[4:8,:,:,:] = im_tensor

                        for batch_im_idx in range(self.dl_studio.batch_size):
                            for mask_layer_idx in range(5):
                                for i in range(64):
                                    for j in range(64):
                                        if mask_layer_idx == 0:
                                            if 25 < outputs[batch_im_idx,mask_layer_idx,i,j] < 85:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 1:
                                            if 65 < outputs[batch_im_idx,mask_layer_idx,i,j] < 135:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 2:
                                            if 115 < outputs[batch_im_idx,mask_layer_idx,i,j] < 185:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 3:
                                            if 165 < outputs[batch_im_idx,mask_layer_idx,i,j] < 230:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 4:
                                            if outputs[batch_im_idx,mask_layer_idx,i,j] > 210:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50

                                display_tensor[8+4*mask_layer_idx+batch_im_idx,:,:,:]= \
                                                          outputs[batch_im_idx,mask_layer_idx,:,:]

                        self.dl_studio.display_tensor_as_image(
                        torchvision.utils.make_grid(display_tensor, nrow=4, normalize=True, 
                                               padding=2, pad_value=10))

#                        plt.imshow(torchvision.utils.make_grid(display_tensor, nrow=4, normalize=True,
#                                               padding=4, pad_value=255))


    def plot_loss(self):
        plt.figure()
        plt.plot(self.LOSS)
        plt.show()


#_________________________  End of DLStudio Class Definition ___________________________

#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    pass
