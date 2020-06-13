# PredictionModel
Deep convolutional neural network to predict COVID-19 in X-Ray images

For the prediction, everything is done using MHCovid class. As simple as the below code:

```python
from MHCovid import MHCovid

mh = MHCovid()
model = mh.generateModel()
print(mh.predict(model, 'covid1.jpg'))
```
