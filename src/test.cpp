/*
    Copyright (c) 2016, Joseph Mariadassou
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * The names of its contributors may be used to endorse or promote products
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
#include <jsonreader/MyCnn.hpp>
using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;
namespace tsp = theSundayProgrammer;
char* data_dir_path = "data";
int main(int argc, char** argv) {
  try
  {
    std::string const  path = argc == 2 ? argv[1] : data_dir_path;
    tsp::MyCNN cnn;
    cnn.GenerateCode(path + "/model.json");
    std::cout << "loaded model" << std::endl;

    std::vector<label_t>  test_labels;
    std::vector<vec_t>  test_images;


    parse_mnist_labels(path + "/t10k-labels.idx1-ubyte", &test_labels);
    parse_mnist_images(path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);
    cnn.LoadData(path + "/model.dat");

    // 
    std::cout << "start testing" << std::endl;
    cnn.test(test_images, test_labels).print_detail(std::cout);
    std::cout << "end training." << std::endl;
  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}


