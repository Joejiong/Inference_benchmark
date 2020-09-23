# Seq2Seq测试

## 一、模型导出

### 1. 静态图模型
Paddle的静态图模型代码在 [models/PaddleNLP/seq2seq/seq2seq/base_model.py]()。`BaseModel`中的`build_graph`方法提供了`mode='beam_search'`参数，用于配置训练模式还是预测模式。

在`base_model.py`同级目录下，新建一个py文件，用于导出静态图模型。代码如下：
```python
import paddle.fluid as fluid
from base_model import BaseModel

def save_inference_model():
    # config hyper parameter
    beam_size = 10
    num_layers = 2
    src_vocab_size = 17191
    tar_vocab_size = 7709
    batch_size = 2
    dropout = 0.0
    init_scale = 0.1
    max_grad_norm = 5.0
    hidden_size = 512
    # instance model
    model = BaseModel(
        hidden_size,
        src_vocab_size,
        tar_vocab_size,
        batch_size,
        num_layers=num_layers,
        init_scale=init_scale,
        beam_max_step_num=50,
        dropout=0.0)

    # build static program
    trans_res = model.build_graph(mode='beam_search', beam_size=beam_size)
    main_program = fluid.default_main_program()
    
    # modify into your path
    model_dir = '/workspace/code_dev/paddle-predict/paddle/static/seq2seq'

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    # save inference
    fluid.io.save_inference_model(model_dir, 
                                feeded_var_names=['src', 'src_sequence_length'], 
                                target_vars=[trans_res], 
                                executor=exe,
                                main_program=main_program,
                                model_filename='model',
                                params_filename='params')

if __name__ == '__main__':
    save_inference_model()
```

执行上述py文件，会在`model_dir`目录下导出静态图模型文件`model`和参数文件`params`.

### 2. 动转静模型
seq2seq动态模型在 [models/dygraph/seq2seq/base_model.py]()路径下。在动转静`to_static`迭代时，我们针对dygraph目录的模型做过对齐。因此动态图模型可以基于 [paddle/python/paddle/fluid/tests/unittests/dygraph_to_static/seq2seq_dygraph_model.py]()路劲代码进行转换。

在`seq2seq_dygraph_model.py`同级目录下，新建一个py文件，代码如下。建议使用paddle2.0beta或develop分支的whl包执行如下代码。

> 注：由于动态图代码写得不是很规范，预测逻辑耦合了batch_size。因此针对不同的batch_size，需要单独保存对应的模型。需修改下述代码中的batch_size值

```python
import paddle
from paddle.static import InputSpec
from paddle.jit import to_static
# paddle.jit.set_verbosity(10)

def save_inference_model():
    
    paddle_model_dir = '/workspace/code_dev/paddle-predict/paddle/dy2stat/'

    paddle.disable_static()
    # need modify this value
    batch_size = 32
    ids_spec = InputSpec(shape=[batch_size, 20], name='src_ids', dtype='int64')
    seq_len_spec = InputSpec(shape=[batch_size], name='src_seq_len', dtype='int64')
    model = BaseModel(
                hidden_size=512,
                src_vocab_size=17191,
                tar_vocab_size=7709,
                batch_size=batch_size, 
                beam_size=10,
                num_layers=2,
                init_scale=0.1,
                dropout=0.0,
                beam_max_step_num=50,
                mode='beam_search')
    
    # bind `beam_search` into `forward`
    model.forward = to_static(model.beam_search, input_spec=[ids_spec, seq_len_spec])

    # ids = paddle.to_tensor(np.random.randint(5000, size=(2, 20)).astype('int64'))
    # seq_len = paddle.to_tensor(np.array([13, 20]).astype('int64'))
    # out = model.beam_search(ids, seq_len)
    # print(out.shape)

    config = paddle.jit.SaveLoadConfig()
    config.model_filename = 'model'
    config.params_filename = 'params'
    paddle.jit.save(model, model_path=paddle_model_dir + 'seq2seq_%s'%batch_size,input_spec=[ids_spec, seq_len_spec], configs=config)
```


**Note:**
此处为了对齐静态图模型的op，因此在`encoding`阶段，需要将动态图代码中的`for`转为`while_op`。为了触发此转换，需要修改`beam_search`中的几行代码：

+ 改动一：
    ```python
    # line 306:
    for k in range(args.max_seq_len):

    # modify into:
    for k in range(max_seq_len):
    ```

+ 改动二：
    ```python
    # line 350,351
    dec_hidden = [self._expand_to_beam_size(ele) for ele in dec_hidden]
    dec_cell = [self._expand_to_beam_size(ele) for ele in dec_cell]

    # modify into:

    dec_hidden = [self._expand_to_beam_size(dec_hidden[idx]) for idx in range(self.num_layers)]
    dec_cell = [self._expand_to_beam_size(dec_cell[idx]) for idx in range(self.num_layers)]
    ```

+ 改动三：
    ```python
    # line 401, 402
    scores = fluid.layers.reshape(
            log_probs, [-1, self.beam_size * self.tar_vocab_size])

    # modify into:
    scores = fluid.layers.reshape(
                    log_probs, [self.batch_size, self.beam_size * self.tar_vocab_size])
    ```


## 二、预测代码
详见`seq2seq.cc`

## 三、性能评估
执行`./re_build.sh`脚本，对`seq2seq.cc`执行编译，在当前目录下会生成一个`seq2seq`可执行文件。

静态图模型预测：
```bash
./seq2seq --use_gpu --batch_size=1 --repeat_time=100 --dirname=your_static_seq2seq_model_path
```

动转静模型预测：
```bash
./seq2seq --use_gpu --batch_size=1 --repeat_time=100 --dirname=your_dy2stat_seq2seq_model_path
```

> 注：动转静模型预测时，不同的batch_size下，要使用对应batch_size的动转静预测模型。