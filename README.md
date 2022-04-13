
* **[Link To The Blog Post](https://srush.github.io/annotated-s4)**


<a href="https://srush.github.io/annotated-s4"><img src="https://user-images.githubusercontent.com/35882/149201164-1723a44a-f34b-467c-94b0-ffda5ebcabbb.png"></a>



## Experiments

### ListOps

Downolad dataset in [GitHub page](https://github.com/google-research/long-range-arena). Unzip the downloaded dataset, and move it to the project folder.

```
python -m s4.train --dataset listops-classification --model s4 --epochs 100 --bsz 50 --d_model 128 --n_layers 4 --ssm_n 64 --lr 1e-2 --p_dropout 0 --lr_schedule
```

Training with the previously defined hyper-parameters yields the test accuracy 54.3% on the test set.

Here is the training curve for ListOps Review.

<img width="1356" alt="jax-imdb" src="https://user-images.githubusercontent.com/13411557/162068429-a62ee0b1-baad-4342-9484-7207e39378f7.jpg">

### IMDB

We used the huggingface datasets for LRA IMDB review task.

```
python -m s4.train --dataset imdb-classification --model s4 --epochs 100 --bsz 50 --d_model 64 --n_layers 4 --ssm_n 64 --lr 1e-2 --p_dropout 0
```

Training with the previously defined hyper-parameters yields the test accuracy 80.7% on the test set.

Here is the training curve for IMDB Review.

<img width="1356" alt="jax-imdb" src="https://user-images.githubusercontent.com/16102460/162067502-f03809c0-0842-4718-a404-859f8d5a1a27.png">
