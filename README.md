# Plagiarism Detection

This project implements a simple Plagiarism Detection web application using Flask, allowing users to check for similarity between provided text and a corpus of texts using various similarity metrics.

## Features

- **Plagiarism Check:** Users can input a text and check for plagiarism against a predefined corpus.
- **Customizable Corpus:** The corpus used for similarity checks can be updated or expanded as needed.
- **Scalable Similarity Metrics:** Utilizes different similarity scoring methods (e.g., cosine similarity, BERT embeddings) for comparison.

## Requirements

- Python 3.x
- Flask
- Transformers (Hugging Face)
- pandas
- scikit-learn

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/i-soumya18/Plagiarism-Detection
    cd plagiarism-Detection
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask app:

    ```bash
    python app.py
    ```

2. Open a web browser and go to `http://localhost:5000` to access the Plagiarism Checker.

3. Enter a text in the input field and click "Check for Plagiarism."

## Screenshots
![Plagirasim Detection](attendance_tracker_screenshot_2.png)
![Plagirasim Detection](attendance_tracker_screenshot_2.png)

## Project Structure

- **app.py:** Main Flask application implementing the web interface and similarity check functionalities.
- **index.html:** HTML template for the input page.
- **result.html:** HTML template for displaying the plagiarism check result.
- **corpus_texts.txt:** Text file containing the corpus of texts for comparison.

## Future Improvements

- Implement additional similarity scoring methods.
- Enhance the user interface with more styling and visuals.
- Incorporate user authentication for accessing the service.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any improvements or fixes.

## License

This project is licensed under the [MIT License](LICENSE).

# Contact
For any inquiries or questions, please contact


[![LinkedIn](https://img.shields.io/badge/LinkedIn-Soumyaranjan%20Sahoo-blue?style=for-the-badge&logo=linkedin)](www.linkedin.com/in/soumya-ranjan-sahoo-b06807248/)
[![Twitter](https://img.shields.io/badge/Twitter-%40soumyaranjan__s-blue?style=for-the-badge&logo=twitter)](https://twitter.com/soumya78948)
[![Instagram](https://img.shields.io/badge/Instagram-%40i_soumya18-orange?style=for-the-badge&logo=instagram)](https://www.instagram.com/i_soumya18/)
[![HackerRank](https://img.shields.io/badge/HackerRank-sahoosoumya24201-brightgreen?style=for-the-badge&logo=hackerrank)](https://www.hackerrank.com/sahoosoumya24201)
