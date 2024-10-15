Brainless Model Exposition
==========================

This is the bare model (the actual neural network object definition) wrap around a FastAPI backend.

It is pretty dumb. It need:

* a transformer method
* a translater method
* a model definition (in json)
* the weights of the model

The choice of this brainless model is to provide a standard way to expose any kind of neural network, provided you have their architecture and trained weights.
