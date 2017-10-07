# Neural Network  
  
Simple Neural Network written in C++11.  
The code still needs some tweaking and cleaning (also commenting).  
Neuron object defines a single Neuron.  
NeuralNetwork object defines the whole set of Neuron.  
"main.cpp" is an example. It trains and uses the network as an XOR operation.  
  
### Random Notes:  
- Switched from double to float for increased precision.  
- Using a Kahan summation to minimize precision loss.  
- Double are written in full precision in the save file.  
- Biases needs more testing but they should behave properly.  
- Overall math formalas must be checked again.  
  
### To Do:  
- Check if the Kahan sum is really needed to keep a good precision on double.
- Try to improve the algorithm (emphazise on faster/lighter code). Maybe review the overall architecture of the code.  
- Try to run the code on a GPU (I need to get my hand on a more recent graphic card)