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

        int channels = 3;
        int height = 224;
        int width = 224;
        int input_num = channels * height * width * batch_size;

        float *input = new float[input_num];
        memset(input, 0.5, input_num * sizeof(float));

        // init im_shape
        int shape_num = 2 * batch_size;
        int *shape = new int[shape_num];
        memset(shape, height, shape_num * sizeof(int));

        auto input_names = predictor->GetInputNames();
        auto input_t = predictor->GetInputTensor(input_names[0]);
        input_t->Reshape({batch_size, channels, height, width});
        input_t->copy_from_cpu(input);

        auto shape_t = predictor->GetInputTensor(input_names[1]);
        shape_t->Reshape({batch_size, 2});
        shape_t->copy_from_cpu(shape);

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