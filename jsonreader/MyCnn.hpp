#pragma once
#ifndef CNN_E692F2A00EDC4069939ECF54036096EB
#define CNN_E692F2A00EDC4069939ECF54036096EB
namespace theSundayProgrammer
{
  enum ELossFn
  {
    ELF_not_found,
    ELF_mse,
    ELF_cross_entropy,
    ELF_cross_entropy_multiclass,
  };
  class MyCNN
  {
    //  friend class boost::serialization::access;
  public:
    MyCNN();
    tiny_cnn::result test(std::vector<tiny_cnn::vec_t>&  test_images, std::vector<tiny_cnn::label_t>&  test_labels);

    tiny_cnn::result ComputeResult(std::string const& data, std::string const& labels);

    void LoadData(std::string const& path);

    void SaveData(std::string const& path) const;

    void GenerateCode(std::string const& path);

    template <class T, class S>
    void train(std::vector<tiny_cnn::label_t> const& train_labels,
      std::vector<tiny_cnn::vec_t> const& train_images,
      int minibatch_size,
      T& on_enumerate_minibatch, S& on_enumerate_epoch);

  private:
    tiny_cnn::network<tiny_cnn::sequential> nn;
    std::unique_ptr<tiny_cnn::optimizer>  optimizer;
    ELossFn  lossFn;
  };
  template <class T, class S>
  inline void MyCNN::train(std::vector<tiny_cnn::label_t> const& train_labels,
    std::vector<tiny_cnn::vec_t> const& train_images,
    int minibatch_size,
    T& on_enumerate_minibatch, S& on_enumerate_epoch)
  {
    if (!optimizer)
      throw std::runtime_error("Optimizer no specified");
    switch (lossFn)
    {
    case ELF_mse:
      nn.train<mse>(*optimizer, train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);
      break;
    case  ELF_cross_entropy:
      nn.train<cross_entropy>(*optimizer, train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);
      break;
    case: ELF_cross_entropy_multiclass:
      nn.train<cross_entropy_multiclass>(*optimizer, train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);
      break;
    case ELF_not_found:
      throw std::runtime_error("invalid loss function");
      break;
    }
  }
  inline  MyCNN::MyCNN() :
    lossFn(ELF_mse) {}

}

namespace std
{
  class not_implemented : public runtime_error
  {
  public:
    not_implemented(string const& err) :runtime_error(err) {}
  };
}
#endif