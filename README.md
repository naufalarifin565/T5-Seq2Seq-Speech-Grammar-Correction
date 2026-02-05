# Implementation of a T5-Based Seq2Seq Model for Speech Transcription and Grammar Correction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)

This repository contains the implementation of a **T5-based Sequence-to-Sequence (Seq2Seq) model** designed for **speech transcription and grammar correction** to support the monitoring of children’s English language learning.

This project integrates state-of-the-art NLP techniques, utilizing **OpenAI Whisper** for Automatic Speech Recognition (ASR) and a fine-tuned **T5 (Text-to-Text Transfer Transformer)** for Grammatical Error Correction (GEC).

---

## 📌 Project Overview

Children learning English often produce spoken sentences with grammatical errors. This project aims to address this by automating the correction process.

**Key Objectives:**
- 🗣️ **Transcribe** children’s speech into text using Whisper.
- ✍️ **Correct** grammatical errors using a fine-tuned T5-based Seq2Seq model.
- 📈 **Support** educators and parents in monitoring language development progress.

---

## 🧠 Model & Tech Architecture

The system operates in a pipeline:
1.  **Input:** Audio speech from children.
2.  **Speech-to-Text:** Processed using **Whisper** to generate initial transcripts.
3.  **Grammar Correction:** The transcripts are fed into a **T5 Model** (fine-tuned on datasets like JFLEG) to produce grammatically correct text.
4.  **Comparison/Evaluation:** Performance is evaluated against ground truth. The repository also includes scripts for **Gemini 2.0 Flash** integration for benchmarking or data labeling.

---

## 📂 Repository Structure

Below is the overview of the files included in this repository:

```text
├── data/ & datasets
│   ├── Grammar Correction.csv      # Main training dataset for GEC
│   └── jfleg_validation.csv        # Validation dataset (JFLEG)
│
├── notebooks/ & scripts
│   ├── WHISPER.ipynb               # Speech-to-text implementation using OpenAI Whisper
│   ├── T5PRETRAINASLI.ipynb        # T5 Model pre-training/fine-tuning pipeline
│   ├── Training_NLP_T5.ipynb       # Main training loop for the Grammar Correction model
│   ├── GEMINI2.0Flash.py           # Integration with Google Gemini 2.0 Flash API
│   ├── Interface_FP_NLP (1).ipynb  # Interactive Interface (Demo) for the system
│   └── Seq2SeqSkenario1-3.ipynb    # Sequence-to-Sequence experimentation scenarios
│
├── scenarios/                      # Data preprocessing for specific test scenarios
│   ├── NLP_Dataset_Skenario_1_.ipynb
│   ├── NLP_Dataset_Skenario_2_.ipynb
│   ├── NLP_Dataset_Skenario_3_.ipynb
│   └── NLP_Dataset_Skenario_5_.ipynb
│
└── README.md                       # Project Documentation
