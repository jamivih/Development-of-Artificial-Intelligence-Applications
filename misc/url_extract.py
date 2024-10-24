from newspaper import Article
from requests.exceptions import RequestException
import re

'''
pip install newspaper3k lxml lxml_html_clean

'''
def main(url):
    main_text = extract_main_text(url)
    main_text = remove_empty_lines(main_text) # removes empty spaces between paragraphs, comment line if not needed

    return main_text

# Function to extract main text from a URL
def extract_main_text(url):
    try:
        # Create an Article object and download the content
        article = Article(url)
        article.download()
        
        # Parse the downloaded content
        article.parse()
        
        # Perform natural language processing (optional, to extract keywords, summary, etc.)
        article.nlp()
        
        return article.text

    except RequestException as e:
        return f"An error occurred: {e}"
    except Exception as e:
        return f"Failed to extract article: {e}"
    
def remove_empty_lines(main_text):
    cleaned_text = re.sub(r'\s+', ' ', main_text)
    
    return cleaned_text


if __name__ == "__main__":
    main()
