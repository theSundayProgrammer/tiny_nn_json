# Neural Network Model Archiver

**Aim**  
We want to run a Neural Network (NN) on a slim device such as Raspberry Pi. 
However training the NN on a slim device will be slow. We want to train the NN on a larger 
machine like a desktop and then transmit the model to a slim device.

**Design**  
We describe the model using JSON. This description is used by the 
trainer to create a NN which can then be trained using training and test samples.
We may want to try try different models to decide on the most appropriate one
and then transmit the model to the slim device. The learnt machine is sent as a
simple text file.

**Libraries Used**  
  * *tiny_cnn* from   https://github.com/nyanp/tiny_cnn
  * *jsoncpp* from  https://github.com/open-source-parsers/jsoncpp

**Status**  
The first task is to read a JSON file that generates code code equivalent to the
following C++ code 
  
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32 in, 5x5 kernel, 1-6 fmaps conv
       << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28 in, 6 fmaps, 2x2 subsampling
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
                                     connection_table(connection, 6, 16)) // with connection-table
       << average_pooling_layer<tan_h>(10, 10, 16, 2)
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120)
       << fully_connected_layer<tan_h>(120, 10);
       
 where connection is a one dimensional array of size 6*16.
 
 

