# RAG Chunking Evaluation

This repository contains code and datasets for evaluating chunking strategies in Retrieval-Augmented Generation (RAG) systems. The project includes various benchmarks, data loaders, and utility functions to facilitate the evaluation process.


## Setup

1. **Clone this repository**
2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Copy `.env.example` to `.env` and fill in the required values.

## Usage

Follow the instructions in the `my_benchmark` notebook to run the proposed chunking evaluation framework. The specific chunking strategies under evaluation are detailed in the `chunking_strategies` notebook.

Each step in the evaluation pipeline generates intermediate results, which are saved in the `data` directory for later review and loading.

The `experimental` directory includes tests for other benchmarks and evaluation frameworks, such as Ragas, Trulens, and Multi-Hop-RAG.
