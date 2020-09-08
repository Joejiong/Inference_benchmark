#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <memory>

using Time = decltype(std::chrono::high_resolution_clock::now());
  Time time() {return std::chrono::high_resolution_clock::now();};

  double time_diff(Time t1, Time t2){
      typedef std::chrono::microseconds ms;
      auto diff = t2 - t1;
      ms counter = std::chrono::duration_cast<ms>(diff);
      return counter.count() / 1000.0;
  }

bool test_predictor_latency(const char* argv[]){
    auto model_path = argv[1];
    int repeat = 100;
    int batch_size = 1;

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()){
      std::cout << "CUDA is available, running on GPU" << std::endl;
      device = torch::kCUDA;
    }

    torch::jit::script::Module module = torch::jit::load(argv[1]);
    module.to(device);

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({batch_size, 3, 224,224}, device));
    // inputs.push_back(torch::ones({20}, device));
    
    // warmup
    at::Tensor output = module.forward(inputs).toTensor();
    

    Time time1 = time();
    for(int i=0; i < repeat; ++i){
      at::Tensor output = module.forward(inputs).toTensor();
    }

    auto time2 = time();
    std::cout << "repeat time: " << repeat  << " , model: " << model_path << std::endl;
    std::cout << "batch: " << batch_size << " , predict cost: " << time_diff(time1, time2) / static_cast<float>(repeat) << " ms." << std::endl;
}

int main(int argc, const char* argv[]) {
      if (argc != 2) {
            std::cerr << "usage: example-app <path-to-exported-script-module>\n";
            return -1;
      }

      test_predictor_latency(argv);

    
}
