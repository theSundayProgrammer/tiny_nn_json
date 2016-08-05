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

#include <json/json.h>
#include <algorithm> // sort
#include <memory>
#include <tiny_cnn/tiny_cnn.h>
#include <jsonreader/MyCnn.hpp>
#include <jsonreader/JsonHelper.hpp>
using namespace std;
using namespace tiny_cnn;
namespace theSundayProgrammer
{
  struct AverageParams
  {
    int width, height, sub_sample, channels;
  };


  AverageParams getAverageParams(Json::Value const& item)
  {
    AverageParams params;
    params.width = item.get("width", 0).asInt();
    params.height = item.get("height", 0).asInt();
    params.channels = item.get("in_channels", 0).asInt();
    params.sub_sample = item.get("sub_sample", 0).asInt();
    return params;
  }

  void HandleAveragePooling(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {

    Json::Value nnType = item["param_type"];
    if (!nnType.empty())
    {
      AverageParams params = getAverageParams(item);
      switch (param_supported(nnType.asString()))
      {
      default:
        throw not_implemented(nnType.asString());
        return;
        break;
        //Todo: Change this cut and paste 
      case  EParamTypes::pIdentity:
        nn << average_pooling_layer<tiny_cnn::activation::identity>
          (params.width, params.height, params.channels, params.sub_sample);
        break;
      case  EParamTypes::Softmax:
      case  EParamTypes::Softmaxwithloss:
        nn << average_pooling_layer<tiny_cnn::activation::softmax>
          (params.width, params.height, params.channels, params.sub_sample);
        break;
      case  EParamTypes::Sigmoid:
      case  EParamTypes::Sigmoidcrossentropyloss:
        nn << average_pooling_layer<tiny_cnn::activation::sigmoid>
          (params.width, params.height, params.channels, params.sub_sample);
        break;
      case  EParamTypes::Relu:
        nn << average_pooling_layer<tiny_cnn::activation::relu>
          (params.width, params.height, params.channels, params.sub_sample);
        break;
      case Tan_hp1m2:
        nn << average_pooling_layer<tiny_cnn::activation::tan_hp1m2>
          (params.width, params.height, params.channels, params.sub_sample);
        break;
      case  EParamTypes::Tanh:
        nn << average_pooling_layer<tiny_cnn::activation::tan_h>
          (params.width, params.height, params.channels, params.sub_sample);
        break;
      }

    }
  }

  
}