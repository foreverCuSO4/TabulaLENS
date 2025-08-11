# TabulaLENS

A sophisticated Python pipeline that uses VLM to extract tables from PDF documents, convert them to structured JSON, and then serialize them into human-readable natural language descriptions.

## Features

-   **Stage 1: Table Extraction:** Converts PDF pages into images and uses a vision-language model (`qwen-vl-max`) to identify and extract tables into a JSON format.
-   **Stage 2: Serialization (Optional):** Takes the extracted JSON and uses a powerful text model (`qwen3-30b-a3b-thinking-2507`) to generate descriptive paragraphs.
-   **Stage 3: Combination:** Merges all extracted JSON and natural language files into two timestamped master files for easy access.
-   **Highly Configurable:** All models, paths, and features are controlled via a central `config.ini` file.
-   **Efficient:** Automatically skips files and pages that have already been processed.

## How to Use

### 1. Prerequisites
- Python 3.8+
- Git

### 2. Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/YourUsername/TabulaLENS.git
cd TabulaLENS