This repository constains the code for training spiking neural networks on DVS-Gesture datasets based on the Izhikevich neuron model.

The code relies on the spikingjelly framwork for processing the neuromorphic dataset. Hence, please install the spikingjelly and pytorch first.

pip install spikingjelly

Directly training using the original Izhikevich model results in poor performance, which is below 20% accuracy. The performance is improved by making some simplifications to the original model, which makes the network trainable.

#Training using LIF model

python classify_dvsg_IZK.py -data_dir ./DVS128Gesture -out_dir ./logs -amp -opt Adam -lr_scheduler CosALR -T_max 64 -epochs 1024

#Training using original IZK model

python classify_dvsg_IZK.py -data_dir ./DVS128Gesture -out_dir ./logs -amp -opt Adam -lr_scheduler CosALR -T_max 64 -epochs 1024 -IZK

#Training using simplified IZK model

python classify_dvsg_IZK.py -data_dir ./DVS128Gesture -out_dir ./logs -amp -opt Adam -lr_scheduler CosALR -T_max 64 -epochs 1024 -IZK -simplified
