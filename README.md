# MEng_Project
This is all of the code for Meng project. The project is to develop a LSTM classifier to identify Denial of service(Dos) attack on a SCADA software testbed

The RNN.py can process the raw data from the testbed and make a LSTM classification. When using the RNN.py you need to change the location of training dataset and the test dataset.

The dostraffic.sh is for generating normal and dos attack traffic.

Modbus_IDS is the IDS structure for SCADA systems. Trained machine learning models in keras could be embedded into this structure as ani virus engine.
