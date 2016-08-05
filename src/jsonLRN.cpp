/*
Copyright (c) 2016, Joseph Mariadassou
All rights reserved

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

#include <algorithm> // sort
#include <memory>
#include <tiny_cnn/tiny_cnn.h>
#include <jsonreader/MyCnn.hpp>
#include <jsonreader/JsonHelper.hpp>
using namespace std;
using namespace tiny_cnn;
namespace theSundayProgrammer
{
  struct LRNParams
  {
    int width, height, local_size, channels;
    double alpha, beta;
  };



  LRNParams getLRNParams(Json::Value const& item)
  {
    LRNParams params;
    params.width = item.get("width", 0).asInt();
    params.height = item.get("height", 0).asInt();
    params.channels = item.get("channels", 0).asInt();
    params.local_size = item.get("local_size", 0).asInt();
    params.alpha = item.get("alpha", 1.0).asDouble();
    params.beta = item.get("beta", 5.0).asDouble();
    return params;
  }



  template<class T>
  void assignLRNLayer(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)

  {
    LRNParams params = getLRNParams(item);
    nn << lrn_layer<T>
      (params.width, params.height, params.local_size, params.channels, params.alpha, params.beta);

  }


  void HandleLRN(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {
    Json::Value nnType = item["param_type"];
    if (!nnType.empty())
    {
      switch (param_supported(nnType.asString()))
      {
      default:
        std::runtime_error(std::string("LRN Param type: " + nnType.asString() + " not supported "));
        return;
        break;
      case  EParamTypes::pIdentity:
        assignLRNLayer<tiny_cnn::activation::identity>(item, nn);
        break;
      case  EParamTypes::Softmax:
      case  EParamTypes::Softmaxwithloss:
        assignLRNLayer<tiny_cnn::activation::softmax>(item, nn);
        break;
      case  EParamTypes::Sigmoid:
      case  EParamTypes::Sigmoidcrossentropyloss:
        assignLRNLayer<tiny_cnn::activation::sigmoid>(item, nn);
        break;
      case  EParamTypes::Relu:
        assignLRNLayer<tiny_cnn::activation::relu>(item, nn);
        break;
      case Tanh:
        assignLRNLayer<tiny_cnn::activation::tan_h>(item, nn);
        break;
      case Tan_hp1m2:
        assignLRNLayer<tiny_cnn::activation::tan_hp1m2>(item, nn);
        break;
      }
    }
  }
  

}