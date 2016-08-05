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

  struct ConvolutionParams
  {
    int width, height, window_size, in_channels, out_channels;
  };



  ConvolutionParams getConvolutionParams(Json::Value const& item)
  {
    ConvolutionParams params;
    params.width = item.get("width", 0).asInt();
    params.height = item.get("height", 0).asInt();
    params.in_channels = item.get("in_channels", 0).asInt();
    params.out_channels = item.get("out_channels", 0).asInt();
    params.window_size = item.get("window_size", 0).asInt();
    return params;
  }

  template<class T>
  void assignConvolutionLayer(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {
    Json::Value connections = item["connections"];
    auto params = getConvolutionParams(item);
    if (connections.empty())
      nn << convolutional_layer<T>
      (params.width, params.height, params.window_size,
        params.in_channels, params.out_channels);
    else
    {
      std::unique_ptr<bool[]> connection(new bool[connections.size()]);
      for (size_t index = 0; index < connections.size(); ++index)
      {
        connection[index] = connections[(int)index].asBool();
      }
      nn << convolutional_layer<T> //tiny_cnn::activation::tan_h
        (params.width, params.height, params.window_size,
          params.in_channels, params.out_channels,
          connection_table(*connection.get(), params.in_channels, params.out_channels));
    }
  }
  void HandleConvolutional(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {
    Json::Value nnType = item["param_type"];
    if (!nnType.empty())
    {
      switch (param_supported(nnType.asString()))
      {
      default:
        std::runtime_error(std::string("Convolution Pooling Param type: " + nnType.asString() + " not supported "));
        return;
        break;
      case  EParamTypes::pIdentity:
        assignConvolutionLayer<tiny_cnn::activation::identity>(item, nn);
        break;
      case  EParamTypes::Softmax:
      case  EParamTypes::Softmaxwithloss:
        assignConvolutionLayer<tiny_cnn::activation::softmax>(item, nn);
        break;
      case  EParamTypes::Sigmoid:
      case  EParamTypes::Sigmoidcrossentropyloss:
        assignConvolutionLayer<tiny_cnn::activation::sigmoid>(item, nn);
        break;
      case  EParamTypes::Relu:
        assignConvolutionLayer<tiny_cnn::activation::relu>(item, nn);
        break;
      case Tanh:
        assignConvolutionLayer<tiny_cnn::activation::tan_h>(item, nn);
        break;
      case Tan_hp1m2:
        assignConvolutionLayer<tiny_cnn::activation::tan_hp1m2>(item, nn);
        break;
      }

    }
  }
}