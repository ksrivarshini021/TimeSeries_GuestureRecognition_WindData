# TimeSeries_GuestureRecognition_WindData
machine learning classification problem for gesture recognition on time series data

## Certain keep in mind for data cleaning

The data that was obtained has each Participant's guestures and the referants that they performed, all in a single csv file. 

1. From the single csv of each Participant's data, crop the time-series data that is of a single referant number and save in supare excel files. That would make the next part of cleaning data easiler. 

2. Use the python script `handVisualizeAll.py` to graph all the movements into 3D plane and use the html script, `master_viewer.html` to visualize and interact with the graphs. 

3. Remember to make sure you record the travel time of the hand or the wrist as my hypothesis is that the Machine Learning model should differentiate between the travel time and the actual guester. There are many ways where a mathametical interpolation could be used for simpler data but the data provided here is uniformly distributed into milliseconds for all the participants so the effetient way to record the travel time is to crop **50 frams before the actual guesture**.
