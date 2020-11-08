from requests import Session
from bs4 import BeautifulSoup
import time
import re
from concurrent.futures import ThreadPoolExecutor
import pickle
import requests
import random
from selenium import webdriver
from itertools import repeat
import numpy as np
import pandas as pd

headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36",
}


# phys.org
def get_news_article(news_link, website):
    if website == 'phys':
        r = requests.get(news_link, headers=headers)
        i = 1

        while r.status_code != 200:
            time.sleep(random.randint(4, 7))
            r = requests.get(news_link, headers=headers)
            print(f'Attempt number {i}')
            i += 1

        news_soup = BeautifulSoup(r.content, 'lxml')
        article = news_soup.find('article', {'class': 'news-article'})

        article_text = re.sub(r'\s+', ' ', article.text.strip())

        time.sleep(random.randint(3, 7))
        return article_text

    elif website == 'nytimes':
        total_article = ''

        url = 'https://www.nytimes.com' + news_link
        try:
            r = requests.get(url, headers=headers)
        except Exception:
            print(url)
            return None

        print(news_link[-1])
        soup = BeautifulSoup(r.content, 'lxml')

        if soup.find('p', {'id': 'article-summary'}):
            article_summary = soup.find('p', {'id': 'article-summary'}).text
            article_summary = re.sub(r'\s+', ' ', article_summary.strip())

            total_article += article_summary

        if soup.find('span', {'class': 'css-16f3y1r e13ogyst0'}):
            small_font = soup.find('span', {'class': 'css-16f3y1r e13ogyst0'}).text
            small_font = re.sub(r'\s+', ' ', small_font.strip())

            total_article += ' ' + small_font

        if soup.find('section', {'class': 'meteredContent css-1r7ky0e'}):
            article_body = soup.find('section', {'class': 'meteredContent css-1r7ky0e'}).text
            article_body = re.sub(r'\s+', ' ', article_body.strip())

            total_article += ' ' + article_body

        if total_article != '':
            return total_article
        else:
            print(url)
            return None


def scrap_phys_org():
    data = []
    with Session() as s:
        for i in range(1, 43):
            url = f"https://phys.org/search/page{i}.html?search=Tesla&h=1&s=0"
            r = s.get(url, headers=headers)

            j = 1
            while r.status_code != 200:
                time.sleep(3)
                r = s.get(url, headers=headers)
                print(f'Attempt number {j}')
                j += 1

            soup = BeautifulSoup(r.content, 'lxml')

            link_class = 'news-link'

            links = soup.findAll('a', {'class': link_class})
            links = map(
                lambda link: link['href'],
                list(links)
            )

            with ThreadPoolExecutor() as executor:
                article_texts = list(executor.map(get_news_article, links, repeat('phys')))
                time.sleep(random.randint(3, 6))

            data += article_texts

            print(i, '- done')

    return data


# nytimes.com
def scrap_nytimes_com():
    with Session() as s:
        url = 'https://www.nytimes.com/search?query=tesla'
        driver = webdriver.Chrome(executable_path='/home/ilolio/Documents/chromedriver')
        driver.get(url)
        for i in range(50):
            driver.find_element_by_css_selector('button[data-testid=search-show-more-button]').click()
            time.sleep(2)
            print(i, '- done')

        soup = BeautifulSoup(driver.page_source, 'lxml')

        # Links
        div_class = 'css-e1lvw9'

        divs = soup.findAll('div', {'class': div_class})
        links = pd.DataFrame(data=list(map(
            lambda div: div.a['href'],
            divs
        )), index=None, columns=None)

        filt = ~(links[0].str.startswith('http') | links[0].str.startswith('/video') | \
                 links[0].str.startswith('/slideshow') | links[0].str.startswith('/interactive'))

        links = links[filt]
        links_indecies = links.index
        links = links.to_numpy().flatten()
        print(len(links))
        print(len(set(list(map(lambda x: x[:30], links)))), end='\n\n')

        # Dates
        dates = soup.findAll('span', {'data-testid': 'todays-date'})[1:]
        dates = np.array(list(map(
            lambda date: date.text, dates
        )))

        dates = dates[links_indecies]

        with ThreadPoolExecutor() as executor:
            article_texts = list(executor.map(get_news_article, links, repeat('nytimes')))

        data = [(text, date) for text, date in zip(article_texts, dates) if text]

    return data


data = scrap_nytimes_com()

# Saving scraped data
with open('../../data/data_nytimes.pickle', 'wb') as f:
    pickle.dump(data, f)
