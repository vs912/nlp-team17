# NLP Team 17 Project

This repository contains the code and resources for the NLP Team 17 project.

## Project Structure

- `data_raw/`: Raw data files
- `data_proc/`: Processed data files
- `src/`: Source code
- `experiments/`: Experimental code and notebooks
- `pre_processing.ipynb`: Jupyter notebook for data preprocessing

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd nlp-team17
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your raw data in the `data_raw/` directory
2. Run the preprocessing notebook: `jupyter notebook pre_processing.ipynb`

## Contributing

1. Create a new branch for your feature: `git checkout -b feature-name`
2. Make your changes and commit them: `git commit -am 'Add some feature'`
3. Push to the branch: `git push origin feature-name`
4. Create a new Pull Request
