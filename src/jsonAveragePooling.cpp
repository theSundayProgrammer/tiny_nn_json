

#include <json/json.h>
#include <algorithm> // sort
#include <fstream>
#include <sstream>
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