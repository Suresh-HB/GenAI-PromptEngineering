"""

@Author: Suresh
@Date: 16-10-2024
@Last Modified by: Suresh
@Last Modified time: 16-10-2024
@Title: Python program to perform Gen AI tasks Inferring and Expanding using Gemini API

"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import csv
import time


def read_reviews(file_path, delimiter="---END OF REVIEW---"):
    """
    Reads the reviews from a text file.
    
    Parameters:
        file_path (str): Path of the text file.
        delimiter (str): Delimiter separating reviews in the file.
        
    Returns:
        list: List of non-empty reviews.
    """
    with open(file_path, 'r') as file:
        reviews = file.read().split(delimiter)
    return [review.strip() for review in reviews if review.strip()]


def extract_review_info(review):
    """
    Extracts product name and review text from a review.

    Parameters:
        review (str): The full review text.
        
    Returns:
        tuple: A tuple containing the product name and review text.
    """
    product = ""
    review_text = ""

    for line in review.split("\n"):
        if line.startswith("Product:"):
            product = line.split(":", 1)[1].strip()
        elif line.startswith("Review:"):
            review_text = line.split(":", 1)[1].strip()
    
    return product, review_text


def analyze_sentiment(review_text, chat_session):

    """
    Analyze the sentiment of a review and generate a reply using the chat session.
    
    Parameters:
        review_text (str): The review text to analyze.
        chat_session: The chat session with the Gemini model.
        
    Return:
        tuple: A tuple containing the sentiment (str) and the generated reply (str).
    """
    sentiment_response = chat_session.send_message(f"Categorize the sentiment in one word of this review in Positive/Negative/Neutral: {review_text}")
    sentiment = sentiment_response.text.lower()

    reply_response = chat_session.send_message(f"Add a 40 words reply to this review: {review_text} as per sentiment {sentiment}")
    reply = reply_response.text

    return sentiment, reply


def guess_product(review_text, chat_session):

    """
    Guess the product category based on the review text.
    
    Parameters:
        review_text (str): The review text to analyze.
        chat_session: The chat session with the Gemini model.
        
    Returns:
        str: The guessed product category.
    """

    product_response = chat_session.send_message(f"Guess the product name based on this review in one word: {review_text}")
    return product_response.text.strip()


def save_to_csv(data, csv_file):

    """
    Save the processed review data to a CSV file.
    
    Parameters:
        data (list): List of processed data (rows) to save.
        csv_file (str): The path to the CSV file.

    Return:
        None        
    """

    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Original Product", "Guessed Product", "Review", "Sentiment", "Reply"])
        writer.writerows(data)


def process_reviews(file_path, csv_file, chat_session, delimiter="---END OF REVIEW---"):

    """
    Process reviews by reading from a file, analyzing them, and saving results to a CSV.
    
    Parameters:
        file_path (str): The path to the text file containing reviews.
        csv_file (str): The path to save the processed reviews.
        chat_session: The chat session with the Gemini model.
        delimiter (str): The delimiter separating reviews in the file.

    Return:
        None        
    """
    reviews = read_reviews(file_path, delimiter)
    data = []

    for review in reviews:
        original_product, review_text = extract_review_info(review)
        guessed_product = guess_product(review_text, chat_session)
        sentiment, reply = analyze_sentiment(review_text, chat_session)

        data.append([original_product, guessed_product, review_text, sentiment, reply])
        time.sleep(2)  
    
    save_to_csv(data, csv_file)


def main():
    try:

        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("API key not found in environment variables.")

        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

    
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        chat_session = model.start_chat(history=[])

        process_reviews(r'C:\Users\Suresh\Desktop\pythonBl\Generative_AI\reviews.txt', 'transformed_reviews.csv', chat_session)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
