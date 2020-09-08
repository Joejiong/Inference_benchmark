#include <gflags/gflags.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <cstring>
#include <numeric>
#include <functional>

#include "paddle/include/paddle_inference_api.h"

namespace paddle {

using paddle::AnalysisConfig;

DEFINE_string(dirname, "./mobilenetv1", "Directory of the inference model.");

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time(){return std::chrono::high_resolution_clock::now(); };

double time_diff(Time t1, Time t2){
    typedef std::chrono::microseconds ms;
    auto diff =  t2 - t1;
    ms counter = std::chrono::duration_cast<ms>(diff);
    return counter.count / 1000.0;
}

void PrepareConfig(AnalysisConfig *config, int batch_size){
    config->SetModel(FLAGS_dirname + "/model", FLAGS_dirname + "/params");
    config->EnableUseGpu(100, 0);
    config->SwitchUseFeedFetchOps(false);
}

bool test_map_cnn(int batch_size, int repeat){
    AnalysisConfig config;
    PrepareConfig(&config, batch_size);

    AnalysisConfig config1;
    PrepareConfig(&config1, batch_size);

    AnalysisConfig config2;
    PrepareConfig(&config2, batch_size);

    PaddlePredictor *pres[3];
    auto pred0 = CreatePaddlePredictor(config);
    auto pred1 = CreatePaddlePredictor(config1);

    pres[0] = pred0.get();
    pres[1] = pred1.get();

    int channels = 3;
    int height = 224;
    int width = 224;
    int input_num = channels * height * width * batch_size;

    std::vector<PaddleTensor> inputs;
    float *input = new float[input_num];
    memset(input, 0, input_num * sizeof(float));

    std::vector<std::thread> threads;
    int num_jobs = 2;
    auto time1 = time();
    for(int tid=0; tid < num_jobs; ++tid){
        threads.emplace_back([&, tid](){
            std::vector<PaddleTensor> local_outputs;
            for(size_t i=0; i < 2000; ++i){
                auto input_names = pres[tid]->GetInputNames();
                auto input_t = pres[tid]->GetInputTensor(input_names[0]);
                input_t->Reshape({batch_size, channels, height, width});
                input_t->copy_from_cpu(input);

                pres[tid]->ZeroCopyRun();

                std::vector<float> out_data;
                auto output_names = pres[tid]->GetOutputNames();
                auto output_t = pres[tid]->GetOutputTensor(output_names[0]);
                std::vector<int> output_shape = output_t->shape();
                int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

                out_data.resize(out_num);
                output_t->copy_to_cpu(out_data.data());

                std::cout << "run: " << tid << " " << out_num << " " << batch_size << std::endl;
            }
        });
    }
    for(int i=0;i<num_jobs;++i){
        threads[i].join();
    }

    auto time2 = time();
    std::cout << "batch: " << batch_size << " predict cost: " << time_diff(time1, time2) / 4000.0 << "ms" << std::endl;
    return true;
}

};

int main(){
    for(int i=0;i<1;++i){
        paddle::test_map_cnn(1<<2,1000);
    }
    return 0;
}