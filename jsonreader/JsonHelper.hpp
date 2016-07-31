#pragma once
#ifndef CNN_E692F2A01EDC4069939ECF54036096EB
#define CNN_E692F2A01EDC4069939ECF54036096EB
#include <string>
#include <tiny_cnn/tiny_cnn.h>
#include <json/json.h>
namespace theSundayProgrammer
{
  enum EOptimizerTypes
  {
    Eadagrad,
    ERMSprop,
    Eadam,
    Emomentum,
    Egradient_descent,
    EONotFound
  };
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

 

  void HandleMaxPooling(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn);
  void HandleAveragePooling(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn);
  void HandleConvolutional(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn);
  void HandleFullyConnected(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn);
  void HandleLRN(
    Json::Value const& item,
    tiny_cnn::network<tiny_cnn::sequential>& nn);

  EParamTypes param_supported(const std::string& type_);
}
#endif
