


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
  

  struct FullyConnectedParams
  {
    int in_nodes, out_nodes;
  };


  FullyConnectedParams getFullyConnectedParams(Json::Value const& item)
  {
    FullyConnectedParams params;
    params.in_nodes = item.get("in_nodes", 0).asInt();
    params.out_nodes = item.get("out_nodes", 0).asInt();
    return params;
  }




  template<class T>
  void assignFullyConnected(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {
    FullyConnectedParams params = getFullyConnectedParams(item);
    nn << fully_connected_layer<T>(params.in_nodes, params.out_nodes);

  }

  void HandleFullyConnected(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {
    Json::Value nnType = item["param_type"];
    if (!nnType.empty())
    {

      switch (param_supported(nnType.asString()))
      {
      default:
        std::runtime_error(std::string("Fully Connected Param type: ") + nnType.asString() + " not supported ");
        return;
      case  EParamTypes::pIdentity:
        assignFullyConnected<tiny_cnn::activation::identity>(item, nn);
        break;
      case  EParamTypes::Softmax:
      case  EParamTypes::Softmaxwithloss:
        assignFullyConnected<tiny_cnn::activation::softmax>(item, nn);
        break;
      case  EParamTypes::Sigmoid:
      case  EParamTypes::Sigmoidcrossentropyloss:
        assignFullyConnected<tiny_cnn::activation::sigmoid>(item, nn);
        break;
      case  EParamTypes::Relu:
        assignFullyConnected<tiny_cnn::activation::relu>(item, nn);
        break;
      case Tanh:
        assignFullyConnected<tiny_cnn::activation::tan_h>(item, nn);
        break;
      case Tan_hp1m2:
        assignFullyConnected<tiny_cnn::activation::tan_hp1m2>(item, nn);
        break;
      }

    }
  }

}