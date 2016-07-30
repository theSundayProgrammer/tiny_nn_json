

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
  struct MaxParams
  {
    int width, height, pooling_size, channels;
  };


  MaxParams getMaxParams(Json::Value const& item)
  {
    MaxParams params;
    params.width = item.get("width", 0).asInt();
    params.height = item.get("height", 0).asInt();
    params.channels = item.get("channels", 0).asInt();
    params.pooling_size = item.get("pooling_size", 0).asInt();
    return params;
  }

  void HandleMaxPooling(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {

    Json::Value nnType = item["param_type"];
    if (!nnType.empty())
    {
      MaxParams params = getMaxParams(item);
      switch (param_supported(nnType.asString()))
      {
      default:
        throw not_implemented(nnType.asString());
        return;
        break;
        //Todo: Change this cut and paste 
      case  EParamTypes::pIdentity:
        nn << max_pooling_layer<tiny_cnn::activation::identity>
          (params.width, params.height, params.channels, params.pooling_size);
        break;
      case  EParamTypes::Softmax:
      case  EParamTypes::Softmaxwithloss:
        nn << max_pooling_layer<tiny_cnn::activation::softmax>
          (params.width, params.height, params.channels, params.pooling_size);
        break;
      case  EParamTypes::Sigmoid:
      case  EParamTypes::Sigmoidcrossentropyloss:
        nn << max_pooling_layer<tiny_cnn::activation::sigmoid>
          (params.width, params.height, params.channels, params.pooling_size);
        break;
      case  EParamTypes::Relu:
        nn << max_pooling_layer<tiny_cnn::activation::relu>
          (params.width, params.height, params.channels, params.pooling_size);
        break;
      case Tan_hp1m2:
        nn << max_pooling_layer<tiny_cnn::activation::tan_hp1m2>
          (params.width, params.height, params.channels, params.pooling_size);
        break;
      case  EParamTypes::Tanh:
        nn << max_pooling_layer<tiny_cnn::activation::tan_h>
          (params.width, params.height, params.channels, params.pooling_size);
        break;
      }

    }
  }

  
}