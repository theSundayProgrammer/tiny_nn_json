


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