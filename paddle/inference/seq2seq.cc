#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <functional>
#include <iostream>
#include <cstring>
#include <algorithm>
#include "paddle/include/paddle_inference_api.h"

DEFINE_string(dirname, "../seq2seq", "Directory of the inference model.");
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

        int max_seq_len = 20;
        int ids_num = batch_size * max_seq_len;

        // src word ids
        int64_t *src_ids = new int64_t[ids_num];
        int64_t constant_id = 12;
        std::fill(src_ids, src_ids+ids_num, constant_id);
        // memset(src_ids, constant_id, ids_num * sizeof(int64_t)); 

        // src seq len
        int64_t *seq_len = new int64_t[batch_size];
        memset(seq_len, max_seq_len, batch_size * sizeof(int64_t));

        auto input_names = predictor->GetInputNames();
        auto input_t = predictor->GetInputTensor(input_names[0]);  // src_ids
        input_t->Reshape({batch_size, max_seq_len});
        input_t->copy_from_cpu(src_ids);

        auto seq_len_t = predictor->GetInputTensor(input_names[1]);  // src_seq_len
        seq_len_t->Reshape({batch_size});
        seq_len_t->copy_from_cpu(seq_len);


        // warmup
        for(int i=0;i<5;++i){
            CHECK(predictor->ZeroCopyRun());
        }

        auto time1 = time();
        for(size_t i=0; i < repeat; ++i){
            CHECK(predictor->ZeroCopyRun());

            std::vector<int64_t> out_data;
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