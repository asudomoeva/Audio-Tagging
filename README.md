# Audio Tagging System with Probabilistic Programming
## Project Overview

### Description
**Goal:** Develop an automatic, general-purpose audio tagging system capable of accurately classifying sound collections for a wide range of real-world environments.

**Data:** The original dataset is taken from Kaggle \citep{kaggle}. The samples (20,000 WAV files) are generated from Freesound's library and include things like musical instruments, domestic sounds, and animals \citep{freesound}. Each input represents a WAV file with a corresponding annotative label. There are 41 labels overall, each generated from Google’s AudioSet ontology. The dataset also includes a boolean column indicating whether the label was manually verified.

### Proposal
To achieve the goal, we will be cycling through Box’s loop. Due to the complexity of the task, we propose two separate stages to address both the model performance (given a fixed number of labels) as well as generalizing to the complexity of real-world data (e.g. classifying sounds that were not in the training set).

**Stage 1:** This stage will focus on tuning the model for the highest possible performance given a fixed number of labels. The test will be performed on a subset of the data with only training labels in place.

Having achieved a high performing model during stage 1, it would still not be representative of the real world (expected poor performance on sounds outside of the original labeling).

**Stage 2:** This stage will focus on using Google’s AudioSet ontology tree to improve the model performance on new sounds (i.e. sounds whose labels were not part of original learning)
