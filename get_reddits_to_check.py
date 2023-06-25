import os
import glob

from dotenv import load_dotenv

from what_is_love import RedditScraper

if __name__ == '__main__':
    load_dotenv()
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent = os.getenv('USER_AGENT')
    subreddits = [
        'love', 'relationships', 'relationship_advice', 'quotes', 'Poetry',
    ]
    scraper = RedditScraper(client_id, client_secret, user_agent, subreddits, num_posts=1000)
    scraper.scrape(output_file="csv_files/reddit_posts_20230503.tsv")
    reddit_posts = glob.glob("csv_files/01_reddit_posts/*.tsv")
