# Exploring Deep Learning

A single Google Colab notebook, [`DEEPLEARNING_LAB.ipynb`](DEEPLEARNING_LAB.ipynb), covering
the Deep Learning lab end to end — from classical ML baselines, through neural networks
built from scratch in NumPy, to CNNs, RNNs, and a Keras NLP classifier. Each section is a
self-contained cell with numbered steps and its own plots.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sania0507/Exploring-Deep-Learning/blob/main/DEEPLEARNING_LAB.ipynb)

## Contents

| Section | Topic | Built with |
| --- | --- | --- |
| Python operators | Complete operator reference: arithmetic, comparison, logical, bitwise | Python |
| PCA | Dimensionality reduction and explained-variance analysis | scikit-learn |
| Linear regression | Metric comparison: MSE, MAE, R² | scikit-learn |
| Logistic regression | Accuracy, confusion matrix, classification report | scikit-learn |
| SVM | Support vectors and margins | scikit-learn |
| Learning curve — Iris | Training vs validation accuracy | scikit-learn |
| Learning curve comparison | All algorithms plotted on one axis | scikit-learn + Keras |
| CNN | MNIST digit classification | TensorFlow / Keras |
| ANN, single layer | From-scratch neural network on Iris | NumPy |
| ANN, multilayer | From-scratch deep network on Iris | NumPy |
| Backpropagation | Deriving and applying the chain rule on XOR | NumPy |
| RNN | Sequence prediction | TensorFlow / Keras |
| NLP | Sentiment classification with tokenization + padding | TensorFlow / Keras |
| Advanced CNN | Deeper convolutional architecture | TensorFlow / Keras |
| RBF network | Radial basis function network on Iris | NumPy |
| ADALINE | Adaptive Linear Neuron on Iris | NumPy |

### Section by section

**PCA.** Standardises the Iris features with `StandardScaler` — mandatory before PCA,
since PCA maximises variance and unscaled features dominate purely by unit size — then
projects to 2 components and reports the new shape, per-component explained variance,
and a scatter of the projected data coloured by class.

**Classical baselines.** Linear regression reports MSE, MAE, and R² together so the
contrast between error metrics and a scale-free goodness-of-fit measure is explicit.
Logistic regression and SVM both produce accuracy, a confusion matrix, and a full
classification report (per-class precision/recall/F1) rather than accuracy alone — worth
seeing on Iris, where one class is linearly separable from the other two and accuracy
alone hides that. The SVM section displays the support vectors themselves.

**Learning curves.** The Iris learning curve plots training accuracy against validation
accuracy as training set size grows, which is the clearest diagnostic for overfitting
(the two curves diverge) versus underfitting (both stay low). The comparison section
then puts linear regression, logistic regression, SVM, single-layer ANN, multilayer
ANN, and CNN on the same axes.

**Neural networks from scratch.** Single-layer, multilayer, backpropagation, RBF, and
ADALINE are implemented in raw NumPy — forward pass, loss, gradient, and weight update
written out explicitly rather than delegated to `model.fit()`. The backpropagation
section works the XOR problem, which a single-layer perceptron provably cannot solve,
making it the natural demonstration of why hidden layers and the chain rule matter.
Each section reports a final loss and renders the learned decision boundary or
learning curve.

**Deep learning with Keras.** The CNN trains on MNIST with normalisation and reshaping
steps shown explicitly, plus a grid of sample images. The advanced CNN extends this to a
deeper architecture. The RNN handles sequence prediction on a synthetic series. The NLP
section builds a sentiment classifier through the full text pipeline: create text data,
tokenize, pad sequences to a uniform length, then train an embedding-based model.

## Running

No local install needed:

1. Click the **Open in Colab** badge.
2. `Runtime → Change runtime type` and confirm **CPU** is selected — everything here runs
   on CPU, no GPU required.
3. Run cells top to bottom. Every dataset used is bundled with scikit-learn, TensorFlow,
   or generated inline, so **there are no uploads to provide**.

To run locally:

```bash
pip install jupyter numpy pandas matplotlib scikit-learn tensorflow
jupyter notebook DEEPLEARNING_LAB.ipynb
```

`tensorflow` is only needed for the CNN, advanced CNN, RNN, and NLP sections; the
from-scratch NumPy implementations run without it.

## Notes

- Expect the first cell that imports TensorFlow to be slow — Colab sometimes re-installs
  or initialises the TF runtime on first use.
- The MNIST download happens automatically on first run of the CNN section and is cached
  in the session afterwards.
- The two "learning curve comparison" sections are near-duplicates; the later one is the
  up-to-date version. Running both is harmless but wastes a few minutes.
- The NumPy sections intentionally set `np.random.seed(...)` so the from-scratch training
  loops converge reproducibly — changing the seed can change whether the XOR demo
  converges within its epoch budget.
- With no cell outputs stored, open the notebook in Colab (rather than relying on
  GitHub's preview) to see the plots.
