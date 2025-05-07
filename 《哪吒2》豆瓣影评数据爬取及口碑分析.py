#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

# 目标URL
url = "https://movie.douban.com/subject/34780991/comments?status=P"

# 设置请求头，模拟浏览器访问
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 发送GET请求
response = requests.get(url, headers=headers)

# 返回状态码
print("状态码:", response.status_code)


# In[2]:


import pandas as pd

# 创建空DataFrame（带表头）
columns = ["用户昵称", "评分", "评论时间", "用户地址", "评论内容"]
df = pd.DataFrame(columns=columns)

# 保存为Excel文件
df.to_excel("comments.xlsx", index=False)


# In[ ]:


import requests
import random
import time
import pandas as pd
from bs4 import BeautifulSoup

# 使用 User-Agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

def get_headers():
    """固定请求头"""
    return {
        'User-Agent': USER_AGENT,
        'Referer': 'https://movie.douban.com/',
        'Host': 'movie.douban.com'
    }

def parse_comment(item):
    """解析单个评论条目"""
    comment = {}
    
    # 用户昵称
    username_tag = item.select_one('.comment-info a')
    comment['用户昵称'] = username_tag.text.strip() if username_tag else ''
    
    # 评分
    rating_tag = item.select_one('span[class^="allstar"]')
    if rating_tag:
        rating_class = rating_tag.get('class', [''])[0]
        comment['评分'] = int(rating_class.replace('allstar', '')) // 10
    else:
        comment['评分'] = '无评分'
    
    # 评论时间
    time_tag = item.select_one('.comment-time')
    comment['评论时间'] = time_tag.get('title').strip() if time_tag else ''
    
    # 用户地址
    location_tag = item.select_one('.comment-location')
    comment['用户地址'] = location_tag.text.strip() if location_tag else ''
    
    # 评论内容
    content_tag = item.select_one('.comment-content')
    comment['评论内容'] = content_tag.text.strip() if content_tag else ''
    
    return comment

def get_comments(url):
    """获取单页评论数据"""
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        items = soup.select('.comment-item')
        return [parse_comment(item) for item in items]
    except Exception as e:
        print(f'请求失败：{url}，错误：{e}')
        return []

def main():
    base_url = 'https://movie.douban.com/subject/34780991/comments'
    all_comments = []
    
    for page in range(5):
        # 构造分页URL
        start = page * 20
        url = f'{base_url}?start={start}&limit=20&status=P&sort=new_score'
        
        # 获取数据
        print(f'正在爬取第 {page + 1} 页...')
        comments = get_comments(url)
        all_comments.extend(comments)
        
        # 随机延时
        sleep_time = random.uniform(3, 5)
        print(f'等待 {sleep_time:.2f} 秒...')
        time.sleep(sleep_time)
    
    # 保存数据，按指定顺序排列字段
    columns_order = ['用户昵称', '评分', '评论时间', '用户地址', '评论内容']
    df = pd.DataFrame(all_comments, columns=columns_order)
    df.to_excel('豆瓣电影评论数据.xlsx', index=False)
    print('数据已保存到 豆瓣电影评论数据.xlsx')

if __name__ == '__main__':
    main()


# In[ ]:


import pandas as pd

df = pd.read_excel('/Users/lxq/Polyu/ML with Python/哪吒豆瓣短评数据爬取/豆瓣电影评论数据.xlsx')
df.head()


# In[ ]:


import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from snownlp import SnowNLP


# In[ ]:


# 读取数据
df = pd.read_excel(r'/Users/lxq/Polyu/ML with Python/哪吒豆瓣影评口碑分析/豆瓣电影评论数据.xlsx')
df = df.dropna(subset=['评论内容'])  # 删除空评论
df['评分'] = pd.to_numeric(df['评分'], errors='coerce')
df.head()


# In[ ]:


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

def get_sentiment(text):
    return SnowNLP(text).sentiments  # 返回0-1的情感值

df['sentiment'] = df['评论内容'].apply(get_sentiment)

# 绘制情感分布直方图
plt.figure(figsize=(10,8))
plt.hist(df['sentiment'], bins=20, color='#4B96E9', edgecolor='black')
plt.title('sentiment score distribution')
plt.xlabel('sentiment score (0:negative ~ 1:positive)')
plt.ylabel('review number')
plt.show()


# In[ ]:


# 自定义停用词表（需准备中文停用词文件）
stopwords = set(open(r'/Users/lxq/Polyu/ML with Python/哪吒豆瓣影评口碑分析/ChineseStopWords.txt', encoding='utf-8').read().split())


# 分词函数
def chinese_word_cut(text):
    return " ".join([word for word in jieba.cut(text) if word not in stopwords and len(word) > 1])

df['cut_comment'] = df['评论内容'].apply(chinese_word_cut)
# 检查 df['cut_comment']
print(df['cut_comment'].dtype)  # 应该是 <class 'object'>


# 提取TF-IDF关键词
tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['cut_comment'])

try:
    keywords = tfidf.get_feature_names_out()  # 尝试新版本方法
except AttributeError:
    keywords = tfidf.get_feature_names()      # 回退到旧版本方法

# 获取前5大关键词
top5_keywords = sorted(zip(keywords, np.asarray(tfidf_matrix.sum(axis=0)).ravel()), 
                      key=lambda x: x[1], reverse=True)[:5]
top5_words = [word[0] for word in top5_keywords]
top5_words


# In[ ]:


# 雷达图
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

scores = []
for word in top5_words:
    mask = df['cut_comment'].str.contains(word)
    scores.append(df[mask]['评分'].mean())

angles = np.linspace(0, 2*np.pi, len(top5_words), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形
scores += scores[:1]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, scores, 'o-', linewidth=2, color='#FF6B6B')
ax.fill(angles, scores, alpha=0.25, color='#FF6B6B')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(top5_words, fontsize=12)
ax.set_title('TOP5关键词对应平均评分雷达图', pad=20)
plt.show()


# In[ ]:


plt.rcParams['font.sans-serif'] = ['Songti SC']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

text = ' '.join(df['cut_comment'])
wordcloud = WordCloud(font_path='/System/Library/Fonts/STHeiti Medium.ttc', 
                     background_color='white',
                     max_words=200,
                     width=1000, height=800).generate(text)

plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('评论词云图')
plt.show()


# In[ ]:


plt.rcParams['font.sans-serif'] = ['Songti SC']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

df['评论时间'] = pd.to_datetime(df['评论时间'])
df.set_index('评论时间', inplace=True)
daily_sentiment = df['sentiment'].resample('D').mean()

plt.figure(figsize=(10,8))
daily_sentiment.plot(color='#4B96E9', marker='o')
plt.title('每日平均情感得分趋势')
plt.xlabel('日期')
plt.ylabel('情感得分')
plt.grid(linestyle='--')
plt.show()

