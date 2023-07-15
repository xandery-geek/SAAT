# Semantic-Aware Adversarial Training for Reliable Deep Hashing Retrieval

This is the PyTorch code of the SAAT paper, which has been accepted by IEEE TIFS.


## Requirements

The code has been tested on PyTorch 1.10. To install the dependencies, run

```shell
pip install -r requirements.txt
```


## Usage

### Training Deep Hashing Models

For example, training a deep hashing model with 32 code length on the NUS-WIDE dataset.
```shell
python hashing.py --train --bit 32 --dataset NUS-WIDE
```

After training, the well-trained models will be saved to `checkpoint` folder. We can evaluate its retrieval performance through MAP metric.

```shell
python hashing.py --test --bit 32 --dataset NUS-WIDE
```

### Adversarial Attack
The following code performs adversarial attack against the well-trained model.

- `attack_method`: name of adversarial attack algorithm, e.g., mainstay (our proposed mainstay codes-based semantic-aware attack).

```shell
python attack.py --bit 32 --dataset NUS-WIDE --attack_method mainstay
```

### Adversarial Training
The following code performs adversarial training for the pre-trained model to alleviate the effects caused by adversarial perturbations.

- `adv_method`: name of adversarial training algorithm, e.g., saat (the proposed semantic-aware adversarial training).

```shell
python defense.py --bit 32 --dataset NUS-WIDE --adv_method saat
```

Then, we again verify the robustness of the model trained with adversarial training.

- `adv`: load model after adversarial training.
- `attack_method`: name of adversarial attack algorithm.
- `adv_method`: name of adversarial training algorithm.

```shell
python attack.py --bit 32 --dataset NUS-WIDE --attack_method mainstay --adv --adv_method saat 
```

## Results

### Comparison with other attack algorithms

![adversarial attck](./figure/adversarial%20attack.jpg?raw=true)


### Precision-Recall curves

![pr curves](./figure/pr%20curves.jpg?raw=true)

### Comparison with other adversarial training methods

![adversarial training](./figure/adversarial%20training.jpg?raw=true)
