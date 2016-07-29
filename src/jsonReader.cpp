

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
//ToDO Lrn
enum EParamTypes
{
  pIdentity,
  Softmaxwithloss,
  Sigmoidcrossentropyloss,
  Relu,
  Sigmoid,
  Tanh,
  Tan_hp1m2,
  Softmax,
  EPNotFound
};
enum EOptimizerTypes
{
  Eadagrad,
  ERMSprop,
  Eadam,
  Emomentum,
  Egradient_descent,
  EONotFound
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
EOptimizerTypes optimizer_supported(const std::string& type_);

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
template<class T>
void assignFullyConnected(
  Json::Value const& item,
  tiny_cnn::network<tiny_cnn::sequential>& nn)
{
  FullyConnectedParams params = getFullyConnectedParams(item);
  nn << fully_connected_layer<T>   (params.in_nodes, params.out_nodes);

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
  std::unique_ptr<tiny_cnn::optimizer> Handleadagrad(Json::Value const& val)
  {
    double alpha = val.get("alpha", 0.03).asDouble();
    std::unique_ptr<tiny_cnn::optimizer> optimiser = make_unique<tiny_cnn::adagrad>();
    tiny_cnn::adagrad& opt = static_cast<tiny_cnn::adagrad&>(*optimiser);
    opt.alpha = alpha;
    return optimiser;
  }


  ELayerTypes layer_supported(const std::string& type_)
  {

    struct 
    {
      string name;
      ELayerTypes id;
    } supported[] = {
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


  EOptimizerTypes optimizer_supported(const std::string& type_)
  {

    struct 
    {
      string name;
      EOptimizerTypes id;
    } supported[] =
    {
      { "adagrad",Eadagrad },
      { "RMSprop",ERMSprop },
      { "adam",Eadam },
      {"gradient_descent",Egradient_descent},
      { "momentum",Emomentum }
    };
    std::string type(type_);
    std::transform(type.begin(), type.end(), type.begin(), tolower);
    for (auto item = begin(supported); item != end(supported); ++item) {
      if (item->name == type) return item->id;
    }
    return EOptimizerTypes::EONotFound;
  }

  EParamTypes param_supported(const std::string& type_) {
  struct 
    {
      string name;
      EParamTypes id;
    }   const supported[] =
    {
      { "identity",pIdentity },
      { "softmax",Softmax },
      { "softmaxwithloss",Softmaxwithloss },
      { "sigmoid",Sigmoid },
      { "sigmoidcrossentropyloss",Sigmoidcrossentropyloss },
      { "relu",Relu },
      { "tanh",Tanh }
    };

    std::string type(type_);
    std::transform(type.begin(), type.end(), type.begin(), tolower);
    for (auto item = begin(supported); item != end(supported); ++item) {
      if (item->name == type) return item->id;
    }
    return EParamTypes::EPNotFound;

  }



namespace theSundayProgrammer
{
  ELossFn loss_function_supported(const std::string& type_) {
    struct 
    { 
      ELossFn id;
      string const& name; 
    } supported[] =
    {
      { ELF_mse, "mse" },
      { ELF_cross_entropy, "cross_entropy" },
      { ELF_cross_entropy_multiclass, "cross_entropy_multiclass" }
    };

    std::string type(type_);
    std::transform(type.begin(), type.end(), type.begin(), tolower);
    for (auto item = begin(supported); item != end(supported); ++item) {
      if (item->name == type) return item->id;
    }
    return ELF_not_found;

  }

  void MyCNN::GenerateCode(std::string const& path)
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
      throw std::runtime_error(std::string("Failed to parse file: ") + reader.getFormattedErrorMessages());
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
          HandleAveragePooling(item, nn);
          break;
        case ELayerTypes::Convolutional:
          HandleConvolutional(item, nn);
          break;
        case ELayerTypes::FullyConnected:
          HandleFullyConnected(item, nn);
          break;
        }
    }
    {
      const Json::Value optimizer_node = root["optimizer"];
      if (!optimizer_node.empty() && !optimizer_node["type"].empty())
      {
        string const opt_type = optimizer_node["type"].asString();
        switch (optimizer_supported(opt_type))
        {
        default:
          throw std::runtime_error(opt_type + ":  optimizer not supported");
          break;
        case Eadagrad:
          optimizer = Handleadagrad(optimizer_node);
          break;
        case ERMSprop:
          throw not_implemented(opt_type);
          break;
        case Eadam:
          throw not_implemented(opt_type);
          break;
        case Emomentum:
          throw not_implemented(opt_type);
          break;
        case Egradient_descent:
          throw not_implemented(opt_type);
          break;
          
        }
      }
    }
    {
      const Json::Value node = root["loss"];
      if (!node.empty() && !node["type"].empty())
      {
        string const type = node["type"].asString();
        this->lossFn = loss_function_supported(type);
      }
    }
  }


  tiny_cnn::result MyCNN::test(std::vector<vec_t>&  test_images, std::vector<label_t>&  test_labels)
  {
    return nn.test(test_images, test_labels);
  }




  void MyCNN::LoadData(std::string const& path)
  {
    std::ifstream ifs(path);
    ifs >> nn;
  }

  void MyCNN::SaveData(std::string const& path) const
  {
    std::ofstream ofs(path);
    ofs << nn;

  }

}