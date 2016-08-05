/*
Copyright (c) 2016, Joseph Mariadassou
All rights reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#ifndef CNN_E692F2A00EDC4069939ECF54036096EB
#define CNN_E692F2A00EDC4069939ECF54036096EB
#include <json/reader.h>
namespace theSundayProgrammer
{
  enum ELossFn
  {
    ELF_not_found,
    ELF_mse,
    ELF_cross_entropy,
    ELF_cross_entropy_multiclass,
  };

  /** \brief CNN is a wrapper around cnn::network to configure a CNN for training or test from a JSON file
  * 
  */
  class CNN
  {
  public:

    /** \brief Default Constructor
    */
    CNN();

    /** \brief Build the CNN from the input JSON DOM object
    */
    void ConstructNN(Json::Value const& input);
    
    /** \brief Test a collection of Input
    */
    tiny_cnn::result test(std::vector<tiny_cnn::vec_t>&  test_images, std::vector<tiny_cnn::label_t>&  test_labels);

    /** \brief Load the built machine
    */
    void LoadData(std::string const& path);

    /** \brief Save the built machine
    */
    void SaveData(std::string const& path) const;

    /** \brief Train the machine. This is a template due to the call back functions
    */
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
  inline void CNN::train(std::vector<tiny_cnn::label_t> const& train_labels,
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
  inline  CNN::CNN() :
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