

#include <json/json.h>
#include <algorithm> // sort
#include <fstream>
#include <sstream>
#include <memory>
#include <tiny_cnn/tiny_cnn.h>
#include "MyCnn.hpp"
using namespace std;
using namespace tiny_cnn;

enum ELayerTypes
{
  Convolutional,
  AveragePooling,
  FullyConnected,
  ELNotFound
};

enum EParamTypes
{
  Innerproduct,
  Convolution,
  Pooling,
  Lrn,
  Dropout,
  Softmaxwithloss,
  Sigmoidcrossentropyloss,
  Relu,
  Sigmoid,
  Tanh,
  Softmax,
  EPNotFound
};

struct LayerXType
{
  string name;
  ELayerTypes id;
};

struct ParamXType
{
  string name;
  EParamTypes id;
};



struct AverageParams
{
  int width, height, sub_sample, channels;
};

struct ConvolutionParams
{
  int width, height, window_size, in_channels, out_channels;
};


struct FullyConnectedParams
{
  int in_nodes, out_nodes;
};


JSONCPP_STRING readInputTestFile(const char* path);

ELayerTypes layer_supported(const std::string& type_);
EParamTypes param_supported(const std::string& type_);


tiny_cnn::network<tiny_cnn::sequential> GenerateCode(string const& path);



class not_implemented : public std::runtime_error
{

public:
  not_implemented(std::string const& err) :std::runtime_error(err) {}
};

JSONCPP_STRING readInputTestFile(const char* path)
{
  std::ifstream ifs(path);
  std::stringstream istr;
  istr << ifs.rdbuf();
  return istr.str();
}


FullyConnectedParams getFullyConnectedParams(Json::Value const& item)
{
  FullyConnectedParams params;
  params.in_nodes = item.get("in_nodes", 0).asInt();
  params.out_nodes = item.get("out_nodes", 0).asInt();
  return params;
}


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
    case  EParamTypes::Innerproduct:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Convolution:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Pooling:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Lrn:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Dropout:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Softmaxwithloss:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Sigmoidcrossentropyloss:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Relu:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Sigmoid:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Tanh:
      nn << average_pooling_layer<tiny_cnn::activation::tan_h>
        (params.width, params.height, params.channels, params.sub_sample);
      break;
    case  EParamTypes::Softmax:
      throw not_implemented(nnType.asString()); break;
    }

  }


}

void HandleConvolutional(
  Json::Value const& item,
  tiny_cnn::network<tiny_cnn::sequential>& nn)
{
  Json::Value nnType = item["param_type"];
  if (!nnType.empty())
  {
    ConvolutionParams params = getConvolutionParams(item);
    switch (param_supported(nnType.asString()))
    {
    default:
      std::runtime_error(std::string("Convolution Pooling Param type: " + nnType.asString() + " not supported "));
      return;
      break;
    case  EParamTypes::Innerproduct:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Convolution:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Pooling:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Lrn:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Dropout:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Softmaxwithloss:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Sigmoidcrossentropyloss:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Relu:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Sigmoid:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Tanh:
    {
      Json::Value connections = item["connections"];
      if (connections.empty())
        nn << convolutional_layer<tiny_cnn::activation::tan_h>
        (params.width, params.height, params.window_size,
          params.in_channels, params.out_channels);
      else
      {
        std::unique_ptr<bool[]> connection(new bool[connections.size()]);
        for (size_t index = 0; index < connections.size(); ++index)
        {
          connection[index] = connections[(int)index].asBool();
        }
        nn << convolutional_layer<tiny_cnn::activation::tan_h>
          (params.width, params.height, params.window_size,
            params.in_channels, params.out_channels,
            connection_table(*connection.get(), params.in_channels, params.out_channels));
      }
    }
      break;
    case  EParamTypes::Softmax:
      throw not_implemented(nnType.asString()); break;
    }

  }


}

void HandleFullyConnected(
  Json::Value const& item,
  tiny_cnn::network<tiny_cnn::sequential>& nn)
{
  Json::Value nnType = item["param_type"];
  if (!nnType.empty())
  {
    FullyConnectedParams params = getFullyConnectedParams(item);

    switch (param_supported(nnType.asString()))
    {
    default:
      std::runtime_error(std::string("Fully Connected Param type: ") + nnType.asString() + " not supported ");
      return;
      break;
    case  EParamTypes::Innerproduct:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Convolution:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Pooling:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Lrn:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Dropout:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Softmaxwithloss:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Sigmoidcrossentropyloss:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Relu:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Sigmoid:
      throw not_implemented(nnType.asString()); break;
    case  EParamTypes::Tanh:
      nn << fully_connected_layer<tiny_cnn::activation::tan_h>
        (params.in_nodes, params.out_nodes);
      break;
    case  EParamTypes::Softmax:
      throw not_implemented(nnType.asString()); break;
    }

  }


}



tiny_cnn::network<tiny_cnn::sequential> MyCNN::GenerateCode(std::string const& path)
{
  int exitCode = 0;
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
    throw std::runtime_error(std::string("Failed to parse file: ") + reader.getFormattedErrorMessages() );
  }


  const Json::Value plugins = root["layers"];
  // Iterate over the sequence elements.
  for (size_t index = 0; index < plugins.size(); ++index)
  {
    Json::Value const& item = plugins[(int)index];
    Json::Value nnType = item["type"];
    if (!nnType.empty())
      switch (layer_supported(nnType.asString()))
      {
      default:
        std::runtime_error(std::string("Item: " + nnType.asString() + " is not supported"));
        break;
      case ELayerTypes::AveragePooling:
        HandleAveragePooling(item,nn);
        break;
      case ELayerTypes::Convolutional:
        HandleConvolutional(item, nn);
        break;
      case ELayerTypes::FullyConnected:
        HandleFullyConnected(item, nn);
        break;
      }
  }
  return nn;

}
ELayerTypes layer_supported(const std::string& type_)
{
  const LayerXType supported[] = {
    { "convolutional", Convolutional },
    { "averagepooling",AveragePooling },
    { "fullyconnected", FullyConnected }
  };
  std::string type(type_);
  std::transform(type.begin(), type.end(), type.begin(), tolower);
  for (auto item = begin(supported); item != end(supported); ++item) {
    if (item->name == type) return item->id;
  }
  return ELayerTypes::ELNotFound;
}

EParamTypes param_supported(const std::string& type_) {
  const ParamXType supported[] =
  {
    { "innerproduct", Innerproduct },
    { "convolution",Convolution },
    { "pooling",Pooling },
    { "lrn",Lrn },
    { "dropout",Dropout },
    { "softmaxwithloss",Softmaxwithloss },
    { "sigmoidcrossentropyloss",Sigmoidcrossentropyloss },
    { "relu",Relu },
    { "sigmoid",Sigmoid },
    { "tanh",Tanh },
    { "softmax",Softmax }
  };

  std::string type(type_);
  std::transform(type.begin(), type.end(), type.begin(), tolower);
  for (auto item = begin(supported); item != end(supported); ++item) {
    if (item->name == type) return item->id;
  }
  return EParamTypes::EPNotFound;

}
