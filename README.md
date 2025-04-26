# Syntax Aligned LMs

## 📚 Overview
This repo contains code for the paper "Deductive Synthesis with Syntax-Aligned Language Models". 
It aims to increase the oversight and control of language models. We scaffold models with a verification and backtracking system to generate code with 100% accuracy

## 💻 Implementation
The code here is for the LLM component of the project.

1. Fine-tune LLMs in PyTorch on a “controllable” programming language.
2. Interface between the model and the verification system
3. Translate between normal code and our controllable language using custom-made parsers.
4. Connect the model to the verifier by creating a backtracker (to “undo” any mistakes made).


## ⚡ Key Contributions
We show that it is possible to greatly enhance model performance on the SPIDER dataset and provide correctness guarantees for every question answered.

