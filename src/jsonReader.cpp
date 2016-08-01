


#include <algorithm> // sort
#include <memory>
#include <tiny_cnn/tiny_cnn.h>
#include <jsonreader/MyCnn.hpp>
#include <jsonreader/JsonHelper.hpp>
using namespace std;
using namespace tiny_cnn;
namespace theSundayProgrammer
{

  enum ELayerTypes
  {
    Convolutional,
    AveragePooling,
    MaxPooling,
    FullyConnected,
    LRN_layer,
    ELNotFound
  };


  

  JSONCPP_STRING readInputTestFile(const char* path);

  ELayerTypes layer_supported(const std::string& type_);
  EOptimizerTypes optimizer_supported(const std::string& type_);

  tiny_cnn::network<tiny_cnn::sequential> GenerateCode(string const& path);



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
      { "maxpooling",MaxPooling },
      { "lrn",LRN_layer },
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
  void HandleLayers(
    Json::Value const& plugins,
    tiny_cnn::network<tiny_cnn::sequential>& nn)
  {
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
        case ELayerTypes::MaxPooling:
          HandleMaxPooling(item, nn);
          break;
        case ELayerTypes::Convolutional:
          HandleConvolutional(item, nn);
          break;
        case ELayerTypes::FullyConnected:
          HandleFullyConnected(item, nn);
          break;
        case ELayerTypes::LRN_layer:
          HandleLRN(item, nn);
          break;
        }
    }

  }


  std::unique_ptr<tiny_cnn::optimizer>
  HandleOptimizer(
    Json::Value const&  optimizer_node )
    {
      if (!optimizer_node.empty() && !optimizer_node["type"].empty())
      {
        string const opt_type = optimizer_node["type"].asString();
        switch (optimizer_supported(opt_type))
        {
        default:
          throw std::runtime_error(opt_type + ":  optimizer not supported");
          break;
        case Eadagrad:
          return Handleadagrad(optimizer_node);
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
      return std::unique_ptr<tiny_cnn::optimizer>();
    }
  ELossFn HandleLossFn(const Json::Value& node)
  {
    if (!node.empty() && !node["type"].empty())
    {
      return loss_function_supported(node["type"].asString());
    }
    else
    {
      return ELossFn::ELF_mse;
    }
  }

  void MyCNN::GenerateCode(Json::Value const& root)
  {
    
    HandleLayers(root["layers"], this->nn);
    this->optimizer = HandleOptimizer(root["optimizer"]);
    this->lossFn = HandleLossFn(root["loss"]);
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
