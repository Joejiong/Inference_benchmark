#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/graph/default_device.h>

#include <chrono>
#include <string>
#include <iostream>
#include <functional>

using namespace std;
typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

using Time = decltype(std::chrono::high_resolution_clock::now());
  Time time() {return std::chrono::high_resolution_clock::now();};

  double time_diff(Time t1, Time t2){
      typedef std::chrono::microseconds ms;
      auto diff = t2 - t1;
      ms counter = std::chrono::duration_cast<ms>(diff);
      return counter.count() / 1000.0;
  }


tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn,
                             std::string checkpoint_fn = "") {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::GraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph
  // tensorflow::graph::SetDefaultDevice("/gpu:3", const_cast<tensorflow::GraphDef*>(&graph_def.graph_def()));
  status = sess->Create(graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // restore model from checkpoint, iff checkpoint is given
  // if (checkpoint_fn != "") {
  //   tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING,
  //                                           tensorflow::TensorShape());
  //   checkpointPathTensor.scalar<tensorflow::tstring>()() = checkpoint_fn;

  //   tensor_dict feed_dict = {
  //       {graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
  //   status = sess->Run(feed_dict, {}, {graph_def.saver_def().restore_op_name()},
  //                      nullptr);
  //   if (status != tensorflow::Status::OK()) return status;
  // } 
  // else {
  //   status = sess->Run({}, {}, {"init"}, nullptr);
  //   if (status != tensorflow::Status::OK()) return status;
  // }

  return tensorflow::Status::OK();
}

int main(int argc, char const *argv[]) {
  const std::string graph_fn = "./frozen_models/resnet50.pb";
  // const std::string checkpoint_fn = "./exported/my_model";

  // prepare session
  tensorflow::Session *sess;
  tensorflow::SessionOptions options;
  TF_CHECK_OK(tensorflow::NewSession(options, &sess));
  
  TF_CHECK_OK(LoadModel(sess, graph_fn));

  // prepare inputs
  tensorflow::TensorShape data_shape({16, 224, 224, 3});
  tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);

  // same as in python file
  // auto data_ = data.flat<float>().data();
  const std::string input_name = "x:0";
  const std::string output_name = "Identity:0";

  tensor_dict feed_dict = {
      {input_name, data},
  };

  std::vector<tensorflow::Tensor> outputs;

  // warmup
  TF_CHECK_OK(
      sess->Run(feed_dict, {output_name}, {}, &outputs));

  int repeat = 1000;
  auto time1 = time();
  for(int i=0; i < repeat; ++i){
    TF_CHECK_OK(
      sess->Run(feed_dict, {output_name}, {}, &outputs));
  }

  auto time2 = time();
  std::cout << "repeat time: " << repeat  << " , model: " << graph_fn << std::endl;
  std::cout << "predict cost: " << time_diff(time1, time2) / static_cast<float>(repeat) << " ms." << std::endl;

  return 0;
}
