from bs4 import BeautifulSoup

def parse_news(html):
    selected_topics = []
    soup = BeautifulSoup(html, 'html.parser')
    date = soup.find("div", {"class": "col-6 col-sm-4 col-xl-4 item-date"})
    article_body = soup.find("div", {"class": "item-text", "itemprop": "articleBody"})
    intro_text = soup.find("p", {"class": "introtext", "itemprop": "description"})
    article_title = soup.find("div", {"class": "item-title"})
    topics = soup.find_all("li", {"class": "breadcrumb-item"})
    for topic in topics:
        if topic.text != 'صفحه اصلی':
            selected_topics.append(topic.text)
    print(date.span.text)
    print(article_body.text)
    print(article_title.h1.text)
    print(intro_text.text)
    print(selected_topics)

    
    # yield {
    #     "title": article_title.h1.text,
    #     "date": date.span.text,
    #     "intro": intro_text.text,
    #     "content": article_body.text,
    #     "topics": selected_topics
    # }

with open("test.html", encoding='utf8') as fp:
    test = fp.read().encode("utf-8")
    parse_news(test)
