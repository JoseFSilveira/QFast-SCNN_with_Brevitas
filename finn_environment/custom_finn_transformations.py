import numpy as np
import qonnx.core.data_layout as DataLayout
import warnings
from onnx import TensorProto
from onnx import helper as oh
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.onnx_exec import execute_node
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

class MoveMulPastAvgPool(Transformation):
    """
    Based on the FINN MoveMulPastMaxPool transformation
    Move non-negative scalar or channelwise mul operations past avg pool operations.
    We want to have muls next to each other such that they can be collapsed into a
    single mul.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "AveragePool"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn(
                            """Mul weight tensor is not set. If it is a constant,
                                please use set_initializer to set the tensor."""
                        )
                        continue
                    avgpool_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    avgpool_in_name = avgpool_node.input[0]
                    avgpool_in_shape = model.get_tensor_shape(avgpool_in_name)
                    ifm_ch = avgpool_in_shape[1]
                    avgpool_out_name = avgpool_node.output[0]
                    avgpool_out_shape = model.get_tensor_shape(avgpool_out_name)

                    # do not support non-2D avgpool
                    kernel_shape = list(get_by_name(avgpool_node.attribute, "kernel_shape").ints)
                    if len(kernel_shape) != 2:
                        continue

                    # do not move negative multiplication factor(s)
                    if (A < 0).any():
                        continue

                    if all(x == 1 for x in A.shape) or A.shape == (1, ifm_ch, 1, 1):
                        # if the mul is scalar or channelwise,
                        # we can simply swap the order of ops
                        # rewire mul input to be avgpool input
                        avgpool_node.input[0] = start_name
                        model.set_tensor_shape(start_name, avgpool_in_shape)
                        model.set_tensor_datatype(start_name, DataType["FLOAT32"])
                        # use old avgpool input tensor as avgpool output
                        avgpool_node.output[0] = avgpool_in_name
                        model.set_tensor_shape(avgpool_in_name, avgpool_out_shape)
                        model.set_tensor_datatype(avgpool_in_name, DataType["FLOAT32"])
                        # use new avgpool output as new mul node input
                        mul_node.input[0] = avgpool_in_name
                        # use old avgpool output as new mul node output
                        mul_node.output[0] = avgpool_out_name
                        model.set_tensor_datatype(avgpool_out_name, DataType["FLOAT32"])
                        # move mul node past avgpool node
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarLinearPastConcat(Transformation):
    """
    Modification of the FINN MoveLinearPastEltwiseAdd transformation.
    Move Add and Mul operations past Concat operations.
    """

    def move_node(self, graph, model, n, prods, node_ind):

        # found! move one of the muls to output, remove the others
        lin0_inputs = [prod.input[0] for prod in prods]
        in0 = n.input[0]
        out = n.output[0]

        # Store the original Concat output input shape to set the shape of the new input tensor after moving the Mul node past the Concat node
        concat_out_shape = model.get_tensor_shape(out)

        # connect the Concat inputs to mul inputs
        for i in range(len(prods)):
            n.input[i] = lin0_inputs[i]

        # connect mul0 output to concat output
        prods[0].output[0] = out

        # Change the input of the future input of mul0 and output of Concat to the original input of Concat
        if concat_out_shape is not None:
            model.set_tensor_shape(in0, concat_out_shape)

        # connect the input of mul0 and output of Concat together
        n.output[0] = in0
        prods[0].input[0] = in0

        # move prods[0] node past Concat node, and remove the others
        for prod in prods:
            graph.node.remove(prod)
        graph.node.insert(node_ind - 2, prods[0])

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Concat":

                # check for tensors on all Concat inputs
                inputs = [input for input in n.input]
                if len(inputs) < 2:
                    continue
                if any(input is None for input in inputs):
                    continue
                in_inits = [model.get_initializer(input) for input in inputs]
                if any(init is not None for init in in_inits):
                    continue

                # check for mul with same initializer on all inputs
                prods = [model.find_producer(input) for input in inputs]

                # check if any branches are empty (i.e., no producer)
                if any(prod is None for prod in prods):
                    continue

                # check if branches come from the same node (i.e., prod0 == prod1)
                ref_prod = prods[0]
                if any(prod == ref_prod for prod in prods[1:]):
                    continue

                # check if all producers inputs are valid (i.e., have at least 2 inputs)
                if any(len(prod.input) < 2 for prod in prods):
                    continue
                inits = [model.get_initializer(prod.input[1]) for prod in prods]

                # if any initializer is None, skip
                if any(init is None for init in inits):
                    continue

                # Check if all initializer are scalars
                if any(init.size != 1 for init in inits):
                    continue

                # check if all producers are Mul and equal or Add and equal. If so move one of the Mul nodes past the Concat and remove the others.
                if all(prod.op_type == "Mul" for prod in prods) or all(prod.op_type == "Add" for prod in prods):
                    ref_init = inits[0]
                    if all(np.array_equal(init, ref_init) for init in inits[1:]):
                        self.move_node(graph, model, n, prods, node_ind)
                        node_ind -= 1
                        graph_modified = True
                else:
                    continue
        model = model.transform(InferShapes())
        return (model, graph_modified)