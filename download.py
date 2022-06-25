import mindspore_hub as mshub
from mindspore import context

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    device_id=0)

model = "mindspore/1.7/fcn8s_voc2012"
network = mshub.load(model)
network.set_train(False)

