This repository constains the code for training spiking neural networks on DVS-Gesture datasets based on the Izhikevich neuron model.

The code relies on the spikingjelly framwork for processing the neuromorphic dataset. Hence, please install the spikingjelly and pytorch first.

pip install spikingjelly

Directly training the original Izhikevich model results in poor performance, which is below 20% accuracy. The performance is improved by making some simplification to the original model and results in a performance improvement to above 90%.

Training using LIF model:\\
python classify_dvsg_IZK.py -data_dir ./DVS128Gesture -out_dir ./logs -amp -opt Adam -lr_scheduler CosALR -T_max 64 -epochs 1024

Training using original IZK model:\\
python classify_dvsg_IZK.py -data_dir ./DVS128Gesture -out_dir ./logs -amp -opt Adam -lr_scheduler CosALR -T_max 64 -epochs 1024 -IZK

Training using simplified IZK model:\\
python classify_dvsg_IZK.py -data_dir ./DVS128Gesture -out_dir ./logs -amp -opt Adam -lr_scheduler CosALR -T_max 64 -epochs 1024 -IZK -simplified
