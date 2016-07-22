/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iostream>
#include <memory>
#include <tiny_cnn/tiny_cnn.h>
#include "MyCnn.hpp"
using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;
char* data_dir_path = "data";
void load(const string& );
void sample2_mlp();
void sample3_dae();
void sample4_dropout();


  
  tiny_cnn::result MyCNN::test(std::vector<vec_t>&  test_images, std::vector<label_t>&  test_labels)
  {
    return nn.test(test_images, test_labels);
  }

 


  void MyCNN::LoadData(std::string const& path)
  {
    std::ifstream ifs(path);
    ifs >> nn;
  }

  void MyCNN::SaveData(std::string const& path) const
  {
    std::ofstream ofs(path);
    ofs<< nn;

  }

int main(int argc, char** argv) {
  try
  {
    std::string const  path = argc == 2 ? argv[1] : data_dir_path;
    MyCNN cnn;
    cnn.GenerateCode(path + "/model.json");
    std::cout << "loaded model" << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels(path + "/train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images(path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(path + "/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

    // 
    std::cout << "start learning" << std::endl;
    int minibatch_size = 10;
    progress_display disp(train_images.size());
    timer t;
    // create callback
    auto on_enumerate_epoch = [&]() {
      std::cout << t.elapsed() << "s elapsed." << std::endl;

      tiny_cnn::result res = cnn.test(test_images, test_labels);

      std::cout << res.num_success << "/" << res.num_total << std::endl;

      disp.restart(train_images.size());
      t.restart();
    };

    auto on_enumerate_minibatch = [&]() {
      disp += minibatch_size;
    };

    
    cnn.train(train_labels, train_images, minibatch_size, on_enumerate_minibatch, on_enumerate_epoch);


    std::cout << "end training." << std::endl;
    cnn.test(test_images, test_labels).print_detail(std::cout);
    cnn.SaveData(path + "/model.dat");
  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}


