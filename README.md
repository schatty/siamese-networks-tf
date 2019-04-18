# Siamese Neural Networks for One-shot Image Recognition
Repository provides nonofficial implementation of Siamese-Networks for the task of one-shot learning in TensorFlow 2.0. Implemenation is based upon _Siamese Neural Networks for One-shot Image Recognition_ paper from Gregory Koch, Richard Zemel and Ruslan Salakhutdinov. Model has been tested on _Omniglot_ dataset.

<img width="1050" alt="omniglot" src="https://user-images.githubusercontent.com/23639048/56372088-fae2bc80-6206-11e9-88fa-cb4d1de806f4.png">

## Dependencies and Installation
* Project has been tested on Ubuntu 18.04 with Python 3.6.8 and TensorFflow 2.0.0-alpha0
* The dependencies are Pillow and tqdm libraries, which are included in setup requirements
* Training and evaluating require `siamnet` lib. Run `python setup.py install` to install it
* To download Omniglot dataset run bash data/download_omniglot.sh from repository's root

## Repository Structure
Repository structured as follows. `siamnet` contains library with the model and data processing-loading procedures. `scripts` contains training and evaluation scripts. `tests` provides minimal tests for training. `results` folder serves as a directory for text logs destination as well as tensorboard data (by default). Also this folder contains .md file with configuration specifications.

## Training and Evaluating
Configuration of training and evaluation procedures is specified by .config files. Most important parameters in config are
- `data.dataset_path` path to the directory with data
- `data.batch` bach size (number of 2-way elements in the batch)
- `data.episodes` number of episodes within each epoch
- `data.cuda` flag to use CUDA acceleration
- `train.epochs` number of epochs to train
- `train.patience` number of allowed epochs without val score improvement
- `train.restore` flag to restore model from existing one (`model.save_dir`)
All the other parameters is less configurable or less significant in current implementation.

To run training procedure run the following command from repository's root
* python scripts/train/run_train.py

To run evaluation procedure run the following command from repository's root
* python scripts/eval/run_eval.py

Training procedure differs from the one presented in original paper. Presented training organized as follows. On the every step two classes are selected randomly. For each class 1 example is chosen randomly two times resulting in two sets of samples to compare within. Thus we have 2 examples of two classes from one side and class-corresponding different samples from another. That combination of true-false pairs are sorted out resulting in 4 vs. 4 samples. That 4x4 samples are multiplied by batch size resulting in [batch * 4] & [batch * 4] "samples vectors" with corresponding labels column (`1` if samples have the same class, `0` otherwise).

## Tests
Basic tests can be launched by following command from root directory (for now tests required GPU support)
* python -m unittest tests/*

## Results
Presented results are different from paper's due to the difference in neural network architecture and data handling. Relatively modest metrics are caused by no hyperparameters search, absence of data transformations and short training procedure and hopefully will be improved by me in future in my spare time. However, model showed prediction capacity and thus can be improved in near future. For the evaluation phase I used batch of size 1 (which means we gather accuracy from each pair of classes) and averaged metric from 1000 trials (episodes).

<img width="365" alt="acc" src="https://user-images.githubusercontent.com/23639048/56374447-d9d09a80-620b-11e9-9ee9-329feabf986d.png">

| Way                 | 5-way     | 10-way    | 20-way    |
|---------------------|-----------|-----------|-----------|
| Accuracy            | 84.9%     | 75.5%     | 66.1%     |

## References
[1] Gregory Koch, Richard Zemel, Ruslan Salakhutdinov _Siamese Neural Networks for One-shot Image Recognition_

[2] Brenden M. Lake, Ruslan Salakhutdinov, Joshua B. Tenenbaum _The Omniglot Challenge: A 3-Year Progress Report_ (https://arxiv.org/abs/1902.03477)
