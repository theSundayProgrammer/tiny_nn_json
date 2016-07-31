


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