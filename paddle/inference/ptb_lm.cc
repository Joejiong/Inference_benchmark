#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <functional>
#include <iostream>
#include <cstring>
#include "paddle/include/paddle_inference_api.h"

DEFINE_string(dirname, "../mobilenetv1", "Directory of the inference model.");
DEFINE_bool(use_gpu, true, "whether use gpu");
DEFINE_bool(use_mkldnn, true, "whether use mkldnn");
DEFINE_int32(batch_size, 1, "batch size of inference model");
DEFINE_int32(repeat_time, 100, "repeat time for inferencing.");

namespace paddle{
    using paddle::AnalysisConfig;

    using Time = decltype(std::chrono::high_resolution_clock::now());
    Time time() {return std::chrono::high_resolution_clock::now();};

    double time_diff(Time t1, Time t2){
        typedef std::chrono::microseconds ms;
        auto diff = t2 - t1;
        ms counter = std::chrono::duration_cast<ms>(diff);
        return counter.count() / 1000.0;
    }

    void PrepareTRTConfig(AnalysisConfig *config){
        config->SetModel(FLAGS_dirname + "/model", FLAGS_dirname + "/params");
        if(FLAGS_use_gpu){
            config->EnableUseGpu(1000, 4); // gpu:4
        }else{
            config->DisableGpu();
            if(FLAGS_use_mkldnn) config->EnableMKLDNN();
        }
        
        config->SwitchUseFeedFetchOps(false);
        config->SwitchIrOptim(false);
    }


    bool test_predictor_latency(){
        int batch_size = FLAGS_batch_size;
        int repeat = FLAGS_repeat_time;
        AnalysisConfig config;
        PrepareTRTConfig(&config);
        auto predictor = CreatePaddlePredictor(config);

        int num_step = 20;
        int input_num = batch_size * num_step;

        int64_t *input = new int64_t[input_num];
        memset(input, 0, input_num * sizeof(int64_t));

        // init init_hidden
        int num_layer = 2;
        int hidden_size = 200;

        int hidden_num = num_layer * hidden_size * batch_size;
        float *init_hidden = new float[hidden_num];
        memset(init_hidden, 0, hidden_num * sizeof(float));
        // init init_hidden
        float *init_cell = new float[hidden_num];
        memset(init_cell, 0, hidden_num * sizeof(float));

        auto input_names = predictor->GetInputNames();
        auto input_t = predictor->GetInputTensor(input_names[0]);  // input
        input_t->Reshape({batch_size, num_step});
        input_t->copy_from_cpu(input);

        auto init_hidden_t = predictor->GetInputTensor(input_names[1]);  // init_hidden
        init_hidden_t->Reshape({num_layer, batch_size, hidden_size});
        init_hidden_t->copy_from_cpu(init_hidden);

        auto init_cell_t = predictor->GetInputTensor(input_names[2]);  // init_cell
        init_cell_t->Reshape({num_layer, batch_size, hidden_size});
        init_cell_t->copy_from_cpu(init_cell);

        // warmup
        for(int i=0;i<5;++i){
            CHECK(predictor->ZeroCopyRun());
        }

        auto time1 = time();
        for(size_t i=0; i < repeat; ++i){
            CHECK(predictor->ZeroCopyRun());

            std::vector<float> out_data;
            auto output_names = predictor->GetOutputNames();
            auto output_t = predictor->GetOutputTensor(output_names[0]);
            std::vector<int> output_shape = output_t->shape();
            int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
            out_data.resize(out_num);
            output_t->copy_to_cpu(out_data.data());
        }
        auto time2 = time();
        std::cout << "repeat time: " << FLAGS_repeat_time  << " , model: " << FLAGS_dirname << std::endl;
        std::cout << "batch: " << batch_size << " , predict cost: " << time_diff(time1, time2) / static_cast<float>(repeat) << " ms." << std::endl;
        
        return true;

    }
};

int main(int argc, char** argv){
    google::ParseCommandLineFlags(&argc, &argv, true);
    paddle::test_predictor_latency();
    return 0;
}