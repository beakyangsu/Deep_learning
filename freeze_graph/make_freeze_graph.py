
from tensorflow.python.tools import freeze_graph
import argparse
import os

def main():
   parser = argparse.ArgumentParser(description='make pb file.')
   parser.add_argument('-i', '--input_graph',
                        help='path to input graph [.pbtxt]', required=True)
   parser.add_argument('-o', '--output_graph',
                        help='path to output [.pb] file', required=True)
   parser.add_argument('-c', '--checkpoint',
                        help='Path to [.ckpt] file', required=True)
   parser.add_argument('-n', '--out_node',
                        help='out_node ', required=True)
   args = parser.parse_args()


   input_graph = os.path.abspath(args.input_graph)
   output_graph = os.path.abspath(args.output_graph)
   checkpoint = os.path.abspath(args.checkpoint)
   out_node = args.out_node

   checkpoint_stat_name = "checkpoint"
   input_graph_name = input_graph 
   output_graph_name = output_graph 
   input_saver_def_path=""
   input_binary=False
   input_checkpoint_path= checkpoint
   output_node_names= out_node 
   restore_op_name="save/restore_all"
   filename_tensor_name="save/Const:0"
   clear_devices=False

   freeze_graph.freeze_graph(input_graph_name, input_saver_def_path, input_binary, input_checkpoint_path, output_node_names, restore_op_name, filename_tensor_name, output_graph_name, clear_devices, "")

if __name__ == '__main__':
  main()
