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
#include <json/reader.h>
#include <fstream>
#include <sstream>

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;
namespace tsp = theSundayProgrammer;
char* data_dir_path = "data";


JSONCPP_STRING readInputTestFile(const char* path)
{
  std::ifstream ifs(path);
  std::stringstream istr;
  istr << ifs.rdbuf();
  return istr.str();
}
Json::Value getDOM(std::string const& path)
{
  JSONCPP_STRING input = readInputTestFile(path.c_str());
  if (input.empty())
  {
    throw std::runtime_error("Empty input file");
  }

  Json::Features mode = Json::Features::strictMode();
  mode.allowComments_ = true;
  Json::Value root;

  Json::Reader reader(mode);
  bool parsingSuccessful = reader.parse(input.data(), input.data() + input.size(), root);
  if (!parsingSuccessful)
  {
    throw std::runtime_error(
      std::string("Failed to parse file: ") +
      reader.getFormattedErrorMessages());
  }
  return root;
}
int main(int argc, char** argv) {
  try
  {
    std::string const  path = argc == 2 ? argv[1] : data_dir_path;


    tsp::CNN cnn;
    Json::Value root = getDOM(path + "/model.json");
    cnn.ConstructNN(root);
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

namespace theSundayProgrammer
{
  void HandleLayers(
    Json::Value const& plugins,
    tiny_cnn::network<tiny_cnn::sequential>& nn);

  std::unique_ptr<tiny_cnn::optimizer>
    HandleOptimizer(
      Json::Value const&  optimizer_node);

  ELossFn HandleLossFn(const Json::Value& node);

  void CNN::ConstructNN(Json::Value const& root)
  {

    HandleLayers(root["layers"], this->nn);
    this->optimizer = HandleOptimizer(root["optimizer"]);
    this->lossFn = HandleLossFn(root["loss"]);
  }


  tiny_cnn::result CNN::test(std::vector<vec_t>&  test_images, std::vector<label_t>&  test_labels)
  {
    return nn.test(test_images, test_labels);
  }

  void CNN::LoadData(std::string const& path)
  {
    std::ifstream ifs(path);
    ifs >> nn;
  }

  void CNN::SaveData(std::string const& path) const
  {
    std::ofstream ofs(path);
    ofs << nn;

  }

}
