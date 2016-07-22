#pragma once
#ifndef CNN_E692F2A00EDC4069939ECF54036096EB
#define CNN_E692F2A00EDC4069939ECF54036096EB
class MyCNN
{
//  friend class boost::serialization::access;
public:
  
  tiny_cnn::result test(std::vector<tiny_cnn::vec_t>&  test_images, std::vector<tiny_cnn::label_t>&  test_labels);
  
  tiny_cnn::result ComputeResult(std::string const& data, std::string const& labels);
  
  void LoadData(std::string const& path);
  
  void SaveData(std::string const& path) const;
  
  tiny_cnn::network<tiny_cnn::sequential> GenerateCode(std::string const& path);
  template <class T, class S>
  void train(std::vector<tiny_cnn::label_t> const& train_labels,
    std::vector<tiny_cnn::vec_t> const& train_images,
    int minibatch_size,
    T& on_enumerate_minibatch, S& on_enumerate_epoch)
  {
    optimizer.alpha = 0.01*std::sqrt(10.0);
    nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, 20, on_enumerate_minibatch, on_enumerate_epoch);
  }

private:
  tiny_cnn::network<tiny_cnn::sequential> nn;
  tiny_cnn::adagrad optimizer;
  std::unique_ptr<bool[]> connection;
};
#endif