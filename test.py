import numpy as np
import sys
import os
import onnx
import onnxruntime

TMP_FOLDER = "./"


def runOnInput(onnx_file, inputs):
    inputs = np.array(inputs, dtype=np.float32)
    print(inputs)
    print("Runing the network " + onnx_file + " on the input: " + str(inputs))

    # copy the onnx file into another file
    destination = TMP_FOLDER+"onnx_"+str(os.getpid()) + ".onnx"
    os.system("cp " + onnx_file + " " + destination)
    print("done copying "+onnx_file+" into "+destination+" .")

    # use the destination
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1

    print("DEBUG: before sess line ")
    sess = onnxruntime.InferenceSession(destination, sess_options=options)
    print("DEBUG: after sess line ")
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    label_name = sess.get_outputs()[0].name
    print("the network requests input of shape " + str(list(input_shape) +
          ", and of name "+input_name+", the output name is "+label_name+"."))
    print("the input recived to get the output of, is of shape " +
          str(list(inputs.shape)))

    print(input_name, input_shape)
    if list(input_shape) != list(inputs.shape):
        raise ValueError("Inputs shape doesn't match network input shape")
    # .astype(np.float32)
    outputs = sess.run([label_name], {input_name: inputs})
    return np.array(outputs, dtype=np.float32)


def runOnRandomInput(onnx_file):
    print("start runOnRandomInput")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1

    print("DEBUG: before sess line ")
    sess = onnxruntime.InferenceSession(onnx_file)
    print("DEBUG: after sess line ")
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    label_name = sess.get_outputs()[0].name
    print(input_name, input_shape)

    inputs = np.random.rand(*input_shape).astype(np.float32)
    print(
        "Runing the network " + onnx_file +
        " on the the random input: " + str(inputs)
    )
    if list(input_shape) != list(inputs.shape):
        raise ValueError("Inputs shape doesn't match network input shape")
    # .astype(np.float32)
    outputs = sess.run([label_name], {input_name: inputs})
    print("The output is: \n" + str(outputs))
    print("end runOnRandomInput")

    return inputs, np.array(outputs, dtype=np.float32)


def main():
    print("*** In main ***")
    onnx_file = TMP_FOLDER + "test_small.onnx"
    print("Workning on "+onnx_file+" and running it on some random input")

    print("Sending to the function runOnRandomInput to check if it runs.")
    rt_inputs, rt_outputs = runOnRandomInput(onnx_file)
    print("The output after running with onnxruntime is: \n" + str(list(rt_outputs)))


if __name__ == "__main__":
    main()
