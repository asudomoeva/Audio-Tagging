# Audio Tagging System with Probabilistic Programming
## Project Overview

### Description
Due to the vastness of sounds we experience in the real world, no reliable automatic general-purpose audio tagging system currently exists. Many people still believe that audio-tagging presents one of the hardest machine learning problems today. We decided to challenge that perspective by tackling audio classification with probabilistic programming.

**Goal:** Develop an automatic, general-purpose audio tagging system capable of accurately classifying sound collections for a wide range of real-world environments.


**Data:** The original dataset is taken from Kaggle [1]. The samples (20,000 WAV files) are generated from Freesound's library and include things like musical instruments, domestic sounds, and animals [2]. Each input represents a WAV file with a corresponding annotative label. There are 41 labels overall, each generated from Google’s AudioSet ontology. The dataset also includes a boolean column indicating whether the label was manually verified.

### Approach
To achieve the goal, we have cycled through Box’s loop [3]. Our journey consisted of 3 different iterations implementing Probabilistic PCA, and two versions of CNN VAE. 

![](final-project/plots/journey.png)

For details around model accuracies, generated samples, etc please consult *final-project* folder.

## References

1. Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier Favory, Jordi Pons, Xavier Serra. General-purpose Tagging of Freesound Audio with AudioSet Labels: Task Description, Dataset, and Baseline. Submitted to DCASE2018 Workshop, 2018. URL:https://arxiv.org/abs/1807.09902
2. Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font, Dmitry Bogdanov, Andrés Ferraro, Sergio Oramas, Alastair Porter, and Xavier Serra. Freesound datasets: a platform for the creation of open audio datasets. In Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR 2017), pp 486-493. Suzhou, China, 2017.
3. Box, G. E. (1976). Science and statistics. Journal of the American Statistical Association, 71(356):791–799.
